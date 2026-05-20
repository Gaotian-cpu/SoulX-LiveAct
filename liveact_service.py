# -*- coding: utf-8 -*-
"""
Live ACT的服务，本工程的入口
"""
import os
import argparse
import threading
import time
import socket
import subprocess
import shutil
import json
import gc
import datetime
import torch
import torch.distributed as dist
import torchaudio
import torchaudio.transforms as T
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template_string, send_from_directory, jsonify, request, render_template, \
    send_file, abort, after_this_request

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
from util_liveact import center_rescale_crop_keep_ratio, get_embedding, get_msk, get_audio_emb, add_audio_to_video
from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from fp8_gemm import FP8GemmOptions, enable_fp8_gemm
import queue
import silero_vad
from silero_vad import get_speech_timestamps
from datetime import timedelta
import errno

# ================= 1. 全局环境与配置 =================

gc.collect()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")
HLS_ROOT = os.path.join(BASE_DIR, "hls_output")
M3U8_NAME = "live.m3u8"
task_queue = queue.Queue()

os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(HLS_ROOT, exist_ok=True)

# 状态变量
streaming_active = False
task_status_map = {}
task_status_lock = threading.Lock()


# ================= 2. 辅助工具函数 =================

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def resample_audio(audio, sr, fps):
    rate = 25 / fps
    y, sr_out = torchaudio.sox_effects.apply_effects_tensor(audio, sr, [["tempo", f"{rate}"]])
    resampler = T.Resample(sr_out, 16000).to(audio.device)
    return resampler(y) * 3.0, 16000


def update_task_status(task_id, **kwargs):
    with task_status_lock:
        if task_id not in task_status_map:
            task_status_map[task_id] = {}
        task_status_map[task_id].update(kwargs)
        task_status_map[task_id]["updated_at"] = time.time()


def get_task_status(task_id):
    with task_status_lock:
        data = task_status_map.get(task_id)
        return dict(data) if data is not None else None


# ================= 3. 分布式推理引擎类 =================

class DistributedVideoEngine:
    def __init__(self, args):
        self.args = args
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.device = self.local_rank
        self.width, self.height = [int(x) for x in args.size.split('*')]
        self.use_dist = self.world_size > 1

        self.video_save_root = os.path.abspath(getattr(args, "video_save_path", "./generated_videos"))
        os.makedirs(self.video_save_root, exist_ok=True)

        if not dist.is_initialized() and self.world_size>1:
            torch.cuda.set_device(self.device)
            dist.init_process_group(backend="nccl", init_method="env://", rank=self.rank, world_size=self.world_size)

        # 多卡时触发：发送python消息，防止长时间不操作，nccl超时异常
        self.control_pg = dist.new_group(backend="gloo") if self.use_dist else None

        if self.world_size>1:
            from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
            init_distributed_environment(rank=self.rank, world_size=self.world_size)
            initialize_model_parallel(sequence_parallel_degree=self.world_size, ring_degree=1,
                                      ulysses_degree=self.world_size)

        # 加载核心生成模型 (Wan2.1)
        if self.world_size > 1:
            from model_liveact.model_memory_sp import WanModel
        else:
            from model_liveact.model_memory import WanModel
        self.wan_i2v_model = WanModel.from_pretrained(args.ckpt_dir, torch_dtype=torch.bfloat16,
                                                      low_cpu_mem_usage=False)
        self.wan_i2v_model = self.wan_i2v_model.to(dtype=torch.bfloat16)

        enable_fp8_gemm(self.wan_i2v_model, options=FP8GemmOptions())
        if args.block_offload:
            for name, child in self.wan_i2v_model.named_children():
                if name != 'blocks':
                    child.to(self.device)
            self.wan_i2v_model.enable_block_offload(
                onload_device=torch.device(f"cuda:{self.device}"),
            )
        else:
            self.wan_i2v_model = self.wan_i2v_model.to(self.device)
        self.wan_i2v_model.freqs = self.wan_i2v_model.freqs.to(self.device)
        self.wan_i2v_model.eval()
        self.wan_i2v_model = torch.compile(self.wan_i2v_model, mode="max-autotune-no-cudagraphs", backend="inductor", dynamic=False)

        # 采样参数
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.timesteps = [torch.tensor([_]).to(self.device, dtype=torch.float32) for _ in
                          [1000.0, 937.5, 833.33333333, 0.0]]

        # 加载辅件 (VAE / CLIP / T5 / Audio)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_rescale_crop_keep_ratio(pil_image, (self.height, self.width))),
            transforms.ToTensor(),
            transforms.Resize((self.height, self.width)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.vae = LightVAE(vae_path=os.path.join(args.ckpt_dir, 'Wan2.1_VAE.pth'), dtype=torch.bfloat16,
                            device=self.device,
                            use_lightvae=False, parallel=(self.world_size > 1))

        self.clip = CLIPModel(
            checkpoint_path=os.path.join(args.ckpt_dir, 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
            tokenizer_path=os.path.join(args.ckpt_dir, 'xlm-roberta-large'), dtype=torch.bfloat16, device=self.device)

        self.text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16,
                                           device='cpu' if args.t5_cpu else self.device,
                                           checkpoint_path=os.path.join(args.ckpt_dir,
                                                                        'models_t5_umt5-xxl-enc-bf16.pth'),
                                           tokenizer_path=os.path.join(args.ckpt_dir, 'google/umt5-xxl'))

        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            args.wav2vec_dir, local_files_only=True, torch_dtype=torch.bfloat16
        ).to(self.device, dtype=torch.bfloat16).eval()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_dir,
                                                                                  local_files_only=True)

        torch.cuda.empty_cache()
        # 初始化KV Cache
        self.blksz_lst = [6, 8]
        self.frame_len = (self.height // (self.patch_size[1] * self.vae_stride[1])) * (
                self.width // (self.patch_size[2] * self.vae_stride[2]))
        kv_cache_tokens = self.frame_len * sum(self.blksz_lst) // self.world_size
        kv_cache_device = self.device
        kv_cache_dtype = torch.float8_e4m3fn if args.fp8_kv_cache else torch.bfloat16
        kv_scale_shape = (1, kv_cache_tokens, 40, 1)
        self.kv_cache = {
                i: {
                    layer_id: {
                        'k': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                        'v': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                        'k_scale': torch.ones(kv_scale_shape, dtype=torch.float32,
                                              device=kv_cache_device) if args.fp8_kv_cache else None,
                        'v_scale': torch.ones(kv_scale_shape, dtype=torch.float32,
                                              device=kv_cache_device) if args.fp8_kv_cache else None,
                        'mean_memory': False,
                        'offload_cache': False,
                        'fp8_kv_cache': args.fp8_kv_cache,
                    }
                    for layer_id in range(40)
                } for i in range(len(self.timesteps) - 1)
            }
        for n in range(40):
            self.wan_i2v_model.blocks[n].self_attn.init_kvidx(self.frame_len, self.world_size)

        # 编译加速
        self.vae.model.eval()
        # self.vae.encode = torch.compile(self.vae.encode)
        self.vae.decode = torch.compile(self.vae.decode)

        # 加载VAD模型，仅rank 0需要，但为了简单可以都加载
        self.vad_model = silero_vad.load_silero_vad(onnx=False)   # 使用 PyTorch 版本

        # 预热
        print("开始预热")
        start_time = time.perf_counter()
        self._warmup()
        print(f"Total Warmup time {time.perf_counter() - start_time:.4f}s")

    def _warmup(self):
        print(f"[Warmup][Rank {self.rank}] start", flush=True)

        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        try:
            with torch.no_grad():
                frame_num_init = (sum(self.blksz_lst) - 1) * 4 + 1
                # 1. 准备假图像
                cond_image = torch.randn(
                    1, 3, 1, self.height, self.width,
                    device=self.device, dtype=torch.bfloat16
                ).clamp_(-1, 1)
                # 2. CLIP
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    clip_context = self.clip.visual(cond_image)

                # 3. 假音频
                dummy_audio = torch.randn(16000 * 6)
                audio_embedding = get_embedding(
                    dummy_audio,
                    self.wav2vec_feature_extractor,
                    self.audio_encoder,
                    device=self.device)
                # 4. init y
                ref_target_masks = torch.ones(
                    3,
                    self.height // self.vae_stride[1],
                    self.width // self.vae_stride[2],
                    device=self.device,
                    dtype=torch.bfloat16)

                video_frames_placeholder = torch.zeros(
                    1,
                    cond_image.shape[1],
                    frame_num_init - cond_image.shape[2],
                    self.height,
                    self.width,
                    device=self.device,
                    dtype=torch.bfloat16)

                padding_frames = torch.concat([cond_image, video_frames_placeholder], dim=2)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = self.vae.encode(padding_frames).to(self.device).unsqueeze(0)

                msk = get_msk(frame_num_init, cond_image, self.vae_stride, self.device)
                y = torch.concat([msk, y], dim=1)

                # 5. prompt
                context = [
                    self.text_encoder(
                        texts="A person is speaking naturally.",
                        device='cpu' if self.args.t5_cpu else self.device
                    )[0].to(self.device, dtype=torch.bfloat16)
                ]

                # 6. 完全按原逻辑跑，只是 iter_total_num = 2
                iter_total_num = 2
                pre_latent = None

                for iteration in range(iter_total_num):
                    audio_start_idx = 0 if iteration == 0 else (iteration - 1) * self.blksz_lst[-1] * self.vae_stride[0]
                    audio_end_idx = audio_start_idx + frame_num_init

                    audio_embs = get_audio_emb(audio_embedding, audio_start_idx, audio_end_idx, self.device)

                    y_cut = y[:, :, :frame_num_init // 4 + 1, ...]
                    f_idx = 0 if iteration == 0 else 1
                    latent = torch.randn(
                        16,
                        self.blksz_lst[f_idx],
                        self.height // self.vae_stride[1],
                        self.width // self.vae_stride[2],
                        dtype=torch.bfloat16,
                        device=self.device
                    )

                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for i in range(len(self.timesteps) - 1):
                            timestep = self.timesteps[i]
                            arg_c = {
                                'context': context,
                                'clip_fea': clip_context,
                                'ref_target_masks': ref_target_masks,
                                'audio': audio_embs,
                                'y': y_cut[:, :, sum(self.blksz_lst[:f_idx]):sum(self.blksz_lst[:f_idx + 1])],
                                'start_idx': sum(self.blksz_lst[:f_idx]) * self.frame_len,
                                'end_idx': sum(self.blksz_lst[:f_idx + 1]) * self.frame_len,
                                'update_cache': iteration > 1
                            }

                            noise_pred = self.wan_i2v_model(
                                [latent],
                                t=timestep,
                                kv_cache=self.kv_cache[i],
                                skip_audio=False if i in [1, 2] else True,
                                **arg_c
                            )[0]

                            dt = (self.timesteps[i] - self.timesteps[i + 1]) / 1000
                            latent = latent + (-noise_pred) * dt[0]

                        if iteration == 0:
                            _videos = self.vae.decode(latent)
                        else:
                            combined_latent = torch.concat([pre_latent[:, -3:], latent], dim=1)
                            _videos = self.vae.decode(combined_latent)[:, :, 9:]

                        pre_latent = latent
                    torch.cuda.synchronize(self.device)
                    print(f"[Warmup][Rank {self.rank}] iteration {iteration + 1}/2 done", flush=True)

                del cond_image, clip_context, dummy_audio, audio_embedding
                del ref_target_masks, video_frames_placeholder, padding_frames
                del y, msk, context, audio_embs, y_cut, latent, pre_latent, _videos
                if 'combined_latent' in locals():
                    del combined_latent
                if 'noise_pred' in locals():
                    del noise_pred

                # torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)

            if dist.is_initialized():
                dist.barrier()

            print(f"[Warmup][Rank {self.rank}] done", flush=True)
        except Exception as e:
            print(f"[Warmup][Rank {self.rank}] failed: {e}", flush=True)
            raise

    def generate_and_push(self, params):
        global streaming_active

        prompt_list = params['prompt_list']
        fps = int(params['fps'])
        img_path = params['img_path']
        audio_path = params['audio_path']
        task_id = params['task_id']
        main_prompt = params['main_prompt']
        stream_with_audio = bool(params.get('stream_with_audio', False))
        output_mode = params.get('output_mode', 'stream')  # 优先使用请求中的

        task_hls_dir = os.path.join(HLS_ROOT, task_id)
        final_video_path = os.path.join(self.video_save_root, f"{task_id}.mp4")

        hls_ffmpeg_process = None
        save_ffmpeg_process = None
        stats = {}

        def close_proc(proc, name="ffmpeg"):
            if proc is None:
                return
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                ret = proc.wait()
                if ret != 0:
                    print(f"[{name}] exited with code {ret}", flush=True)
            except Exception as e:
                print(f"[{name}] wait failed: {e}", flush=True)

        def write_chunk_bytes(proc, chunk_bytes, name="ffmpeg"):
            if proc is None or proc.stdin is None:
                return
            try:
                proc.stdin.write(chunk_bytes)
                proc.stdin.flush()
            except BrokenPipeError:
                raise RuntimeError(f"{name} stdin broken pipe")
            except Exception as e:
                raise RuntimeError(f"write to {name} failed: {e}")

        def tensor_chunk_to_rgb_bytes(video_tensor):
            """
            video_tensor: [1, 3, T, H, W] in [-1, 1]
            return:
                chunk_bytes: 整个 chunk 的连续 rgb24 bytes
                num_frames: 这个 chunk 的帧数
            """
            video_u8 = (
                ((video_tensor.squeeze(0).permute(1, 2, 3, 0) + 1.0) * 127.5)
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .contiguous()
                    .cpu()
            )  # [T, H, W, C], uint8
            num_frames = video_u8.shape[0]
            chunk_bytes = video_u8.numpy().tobytes()
            return chunk_bytes, num_frames

        try:
            if self.rank == 0:
                update_task_status(
                    task_id,
                    status="running",
                    stage="preparing",
                    message="开始预处理",
                    generated_chunks=0,
                    is_done=False,
                    error=None,
                    stream_ready=False,
                )

            # 1. 音频特征预处理
            if self.rank == 0:
                start_time = time.perf_counter()

            audio_ori, sr_ori = torchaudio.load(audio_path)

            # ========== VAD 处理：检测人声区间 ==========
            # 转换为单声道 numpy 数组 (silero-vad 要求 16kHz 单声道)
            audio_mono = audio_ori.mean(dim=0) if audio_ori.shape[0] > 1 else audio_ori[0]
            # 重采样到 16kHz（如果需要）
            if sr_ori != 16000:
                resampler_vad = T.Resample(sr_ori, 16000)
                audio_16k = resampler_vad(audio_mono.unsqueeze(0)).squeeze(0)
            else:
                audio_16k = audio_mono
            audio_16k_np = audio_16k.cpu().numpy()

            # 获取语音时间戳（单位：秒）
            speech_timestamps = get_speech_timestamps(
                audio_16k_np,
                model=self.vad_model,
                sampling_rate=16000,
                threshold=0.3,                # 敏感度，可调
                min_speech_duration_ms=100,
                min_silence_duration_ms=100,
            )

            # 计算总帧数 (fps 是目标帧率)
            total_frames = int(audio_ori.size(1) / sr_ori * fps)

            # 初始化帧级别的人声标记
            is_speech_frame = [False] * total_frames
            for seg in speech_timestamps:
                start_sec = seg['start'] / 16000.0
                end_sec = seg['end'] / 16000.0
                start_frame = int(start_sec * fps)
                end_frame = int(end_sec * fps)
                for f in range(start_frame, min(end_frame, total_frames)):
                    is_speech_frame[f] = True
            # 可选：保存到 self 或 params 中，供生成循环使用
            # 注意：这里需要将 is_speech_frame 传递给后续循环，可以保存在一个变量中
            # ========== VAD 处理结束 ==========

            audio_resampled, _ = resample_audio(audio_ori, sr_ori, fps)
            audio_embedding = get_embedding(
                audio_resampled[0],
                self.wav2vec_feature_extractor,
                self.audio_encoder,
                device=self.device
            )
            audio_len_sec = audio_ori.size(1) / sr_ori

            if self.rank == 0:
                stats['audio_proc'] = time.perf_counter() - start_time
                update_task_status(task_id, stage="audio_ready", message="音频加载完成")

            # 2. Rank0 启动 ffmpeg
            if self.rank == 0:
                start_time = time.perf_counter()

                if os.path.exists(task_hls_dir):
                    shutil.rmtree(task_hls_dir)
                os.makedirs(task_hls_dir, exist_ok=True)

                if os.path.exists(final_video_path):
                    os.remove(final_video_path)

                # ---------- HLS ffmpeg ----------
                hls_ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',
                    '-loglevel', 'warning',

                    # rawvideo input
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'rgb24',
                    '-s', f'{self.width}x{self.height}',
                    '-r', str(fps),
                    '-i', 'pipe:0',
                ]

                if stream_with_audio:
                    hls_ffmpeg_cmd += [
                        '-thread_queue_size', '1024',
                        '-i', audio_path,
                        '-map', '0:v:0',
                        '-map', '1:a:0',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-af', 'aresample=async=1:first_pts=0',
                        '-shortest',
                    ]
                else:
                    hls_ffmpeg_cmd += [
                        '-an',
                        '-map', '0:v:0',
                    ]

                hls_ffmpeg_cmd += [
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-tune', 'zerolatency',

                    # 固定 1 秒一个关键帧，方便 HLS 切片
                    '-g', str(fps),
                    '-keyint_min', str(fps),
                    '-sc_threshold', '0',

                    '-f', 'hls',
                    '-hls_time', '1',
                    '-hls_list_size', '5',
                    '-hls_segment_type', 'mpegts',
                    '-hls_flags', 'delete_segments+append_list+independent_segments',
                    os.path.join(task_hls_dir, M3U8_NAME)
                ]

                print(f"[Generate][{task_id}] hls_ffmpeg_cmd = {' '.join(map(str, hls_ffmpeg_cmd))}", flush=True)

                # ---------- 保存 mp4 ffmpeg ----------
                # 直接带音频保存，不再先写 silent.mp4 再二次 mux
                save_ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',
                    '-loglevel', 'warning',

                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'rgb24',
                    '-s', f'{self.width}x{self.height}',
                    '-r', str(fps),
                    '-i', 'pipe:0',

                    '-thread_queue_size', '1024',
                    '-i', audio_path,

                    '-map', '0:v:0',
                    '-map', '1:a:0',

                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-af', 'aresample=async=1:first_pts=0',
                    '-shortest',
                    '-movflags', '+faststart',
                    final_video_path
                ]

                print(f"[Generate][{task_id}] save_ffmpeg_cmd = {' '.join(map(str, save_ffmpeg_cmd))}", flush=True)

                if output_mode == "stream":
                    # 原有流式推流代码：启动 hls_ffmpeg_process 和 save_ffmpeg_process
                    hls_ffmpeg_process = subprocess.Popen(
                        hls_ffmpeg_cmd,
                        stdin=subprocess.PIPE,
                        bufsize=0
                    )
                    save_ffmpeg_process = subprocess.Popen(
                        save_ffmpeg_cmd,
                        stdin=subprocess.PIPE,
                        bufsize=0
                    )
                else:
                    # file 模式：不启动任何 ffmpeg 进程，仅准备一个列表收集视频块
                    all_video_chunks = []   # 用于收集每个 chunk 的 tensor
                    hls_ffmpeg_process = None
                    save_ffmpeg_process = None

                stats['ffmpeg_proc'] = time.perf_counter() - start_time
                update_task_status(task_id, stage="ffmpeg_ready", message="推流器已启动")

            # 3. 图像 / 条件
            if self.rank == 0:
                start_time = time.perf_counter()

            image = Image.open(img_path).convert("RGB")
            cond_image = self.transform(image).unsqueeze(1).unsqueeze(0).to(self.device, torch.bfloat16)

            if self.rank == 0:
                stats['image_proc'] = time.perf_counter() - start_time

            if self.rank == 0:
                start_time = time.perf_counter()

            with torch.no_grad():
                clip_context = self.clip.visual(cond_image)

            if self.rank == 0:
                stats['clip_proc'] = time.perf_counter() - start_time

            if self.rank == 0:
                start_time = time.perf_counter()

            torch.manual_seed(self.args.seed)
            ref_target_masks = torch.ones(
                3,
                self.height // self.vae_stride[1],
                self.width // self.vae_stride[2],
                device=self.device,
                dtype=torch.bfloat16
            )
            frame_num_init = (sum(self.blksz_lst) - 1) * 4 + 1
            msk = get_msk(frame_num_init, cond_image, self.vae_stride, self.device)
            video_frames_placeholder = torch.zeros(
                1,
                cond_image.shape[1],
                frame_num_init - cond_image.shape[2],
                self.height,
                self.width,
                device=self.device,
                dtype=torch.bfloat16
            )
            padding_frames = torch.concat([cond_image, video_frames_placeholder], dim=2)
            y = self.vae.encode(padding_frames).to(self.device).unsqueeze(0)
            y = torch.concat([msk, y], dim=1)

            if self.rank == 0:
                stats['init_y'] = time.perf_counter() - start_time

            if self.rank == 0:
                start_time = time.perf_counter()

            edit_prompts = {}
            if prompt_list:
                for edit_prompt in prompt_list:
                    key = (edit_prompt[0], edit_prompt[1])
                    edit_prompts[key] = [
                        self.text_encoder(
                            texts=edit_prompt[2],
                            device='cpu' if self.args.t5_cpu else self.device
                        )[0].to(self.device, dtype=torch.bfloat16)
                    ]

            context_0 = [
                self.text_encoder(
                    texts=main_prompt,
                    device='cpu' if self.args.t5_cpu else self.device
                )[0].to(self.device, dtype=torch.bfloat16)
            ]

            if self.rank == 0:
                stats['prompt_init'] = time.perf_counter() - start_time

            print("\n" + "=" * 30)
            print(f"Task {task_id} Pre-processing Report:")
            for stage, duration in stats.items():
                print(f" - {stage:20}: {duration:.4f}s")
            print("=" * 30 + "\n")

            # ========== 新增：定义动作 prompt（用于无人声段） ==========
            # 添加动作 prompt（用于无人声段）
            action_prompt = "a person smiling naturally, looking around, mouth closed, breathing gently"
            action_context = [
                self.text_encoder(
                    texts=action_prompt,
                    device='cpu' if self.args.t5_cpu else self.device
                )[0].to(self.device, dtype=torch.bfloat16)
            ]
            # ===================================================

            # 4. 主循环
            iter_total_num = int(audio_len_sec / (self.vae_stride[0] * self.blksz_lst[-1] / fps)) + 1
            pre_latent = None

            if self.rank == 0:
                update_task_status(
                    task_id,
                    status="running",
                    stage="generating",
                    message=f"计划生成 {iter_total_num} 个 chunk",
                    total_chunks=iter_total_num,
                    generated_chunks=0,
                    is_done=False,
                )

            for iteration in range(iter_total_num):
                if self.rank == 0:
                    start_time = time.perf_counter()

                cached_context = context_0
                if prompt_list:
                    for k, v in edit_prompts.items():
                        if k[0] <= iteration <= k[1]:
                            cached_context = v
                            break

                audio_start_idx = 0 if iteration == 0 else (iteration - 1) * self.blksz_lst[-1] * self.vae_stride[0]
                audio_end_idx = audio_start_idx + frame_num_init

                # ========== VAD 检测：当前 chunk 是否包含人声 ==========
                start_sec = audio_start_idx / sr_ori
                end_sec = audio_end_idx / sr_ori
                has_speech = False
                for seg in speech_timestamps:
                    seg_start = seg['start'] / 16000.0
                    seg_end = seg['end'] / 16000.0
                    if max(start_sec, seg_start) < min(end_sec, seg_end):
                        has_speech = True
                        break

                force_skip_audio = False

                # if not has_speech and iteration > 0:
                #     # 静音段且不是第一个 chunk：不调用模型，直接生成静态帧
                #     if self.rank == 0:
                #         # 获取上一 chunk 的最后一帧
                #         if all_video_chunks:
                #             last_frame = all_video_chunks[-1][:, :, -1:, :, :]   # [1,3,1,H,W]
                #         else:
                #             last_frame = cond_image.unsqueeze(0)                 # [1,3,1,H,W]
                #         # 当前 chunk 的帧数（非首块固定为 32 帧）
                #         chunk_frames = self.blksz_lst[1] * self.vae_stride[0]   # 8*4=32
                #         _videos = last_frame.repeat(1, 1, chunk_frames, 1, 1)
                #
                #         all_video_chunks.append(_videos.cpu())
                #         update_task_status(
                #             task_id,
                #             status="running",
                #             stage="generating",
                #             message=f"已生成 {iteration + 1}/{iter_total_num} 个 chunk (静音段)",
                #             total_chunks=iter_total_num,
                #             generated_chunks=iteration + 1,
                #             is_done=False,
                #         )
                #         print(
                #             f"静音段 {iteration + 1}/{iter_total_num}, "
                #             f"跳过推理，使用静态帧，耗时:{time.perf_counter() - start_time:.4f}s",
                #             flush=True
                #         )
                #     # 跳过后续的音频 embedding 生成和模型推理
                #     continue
                if not has_speech and iteration > 0:
                    force_skip_audio = True
                    cached_context = action_context   # 替换为动作描述
                    if self.rank == 0:
                        print(f"无人声段 {iteration+1}/{iter_total_num}，使用动作 prompt 并忽略音频")
                # =============== VAD检测 ===============================

                audio_embs = get_audio_emb(audio_embedding, audio_start_idx, audio_end_idx, self.device)

                y_cut = y[:, :, :frame_num_init // 4 + 1, ...]
                f_idx = 0 if iteration == 0 else 1

                latent = torch.randn(
                    16,
                    self.blksz_lst[f_idx],
                    self.height // self.vae_stride[1],
                    self.width // self.vae_stride[2],
                    dtype=torch.bfloat16,
                    device=self.device
                )

                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    for i in range(len(self.timesteps) - 1):
                        timestep = self.timesteps[i]
                        arg_c = {
                            'context': cached_context,
                            'clip_fea': clip_context,
                            'ref_target_masks': ref_target_masks,
                            'audio': audio_embs,
                            'y': y_cut[:, :, sum(self.blksz_lst[:f_idx]):sum(self.blksz_lst[:f_idx + 1])],
                            'start_idx': sum(self.blksz_lst[:f_idx]) * self.frame_len,
                            'end_idx': sum(self.blksz_lst[:f_idx + 1]) * self.frame_len,
                            'update_cache': iteration > 1
                        }

                        # 修改 skip_audio 参数
                        if force_skip_audio:
                            skip_audio = True
                        else:
                            skip_audio = False if i in [1, 2] else True

                        noise_pred = self.wan_i2v_model(
                            [latent],
                            t=timestep,
                            kv_cache=self.kv_cache[i],
                            # skip_audio=False if i in [1, 2] else True,
                            skip_audio=skip_audio,
                            **arg_c
                        )[0]
                        dt = (self.timesteps[i] - self.timesteps[i + 1]) / 1000
                        latent = latent + (-noise_pred) * dt[0]

                    if iteration == 0:
                        _videos = self.vae.decode(latent)
                    else:
                        combined_latent = torch.concat([pre_latent[:, -3:], latent], dim=1)
                        _videos = self.vae.decode(combined_latent)[:, :, 9:]

                    pre_latent = latent

                if self.rank == 0:
                    if output_mode == "stream":
                        # 原有流式逻辑：整个 chunk 一次写入管道
                        # 这里改成“整个 chunk 一次写入”
                        chunk_bytes, num_frames_this_chunk = tensor_chunk_to_rgb_bytes(_videos)

                        write_chunk_bytes(hls_ffmpeg_process, chunk_bytes, name="hls_ffmpeg")
                        write_chunk_bytes(save_ffmpeg_process, chunk_bytes, name="save_ffmpeg")

                        m3u8_path = os.path.join(task_hls_dir, M3U8_NAME)
                        update_task_status(
                            task_id,
                            status="running",
                            stage="generating",
                            message=f"已生成 {iteration + 1}/{iter_total_num} 个 chunk",
                            total_chunks=iter_total_num,
                            generated_chunks=iteration + 1,
                            is_done=False,
                            stream_ready=os.path.exists(m3u8_path),
                        )

                        print(
                            f"生成完成 {iteration + 1}/{iter_total_num}, "
                            f"frames={num_frames_this_chunk}, "
                            f"一个chunk耗时:{time.perf_counter() - start_time:.4f}s",
                            flush=True
                        )
                    else:
                        # 文件模式：收集视频张量到列表，稍后统一保存
                        all_video_chunks.append(_videos.cpu())   # 移到 CPU 避免显存累积

                        # 仍可更新进度（不含 stream_ready）
                        update_task_status(
                            task_id,
                            status="running",
                            stage="generating",
                            message=f"已生成 {iteration + 1}/{iter_total_num} 个 chunk",
                            total_chunks=iter_total_num,
                            generated_chunks=iteration + 1,
                            is_done=False,
                        )

                        print(
                            f"生成完成 {iteration + 1}/{iter_total_num}, "
                            f"一个chunk耗时:{time.perf_counter() - start_time:.4f}s (文件模式)",
                            flush=True
                        )

            # 5. 收尾
            if self.rank == 0:
                if output_mode == "stream":
                    update_task_status(
                        task_id,
                        status="running",
                        stage="finalizing",
                        message="视频生成完成，正在封装最终文件",
                        is_done=False,
                    )

                    close_proc(hls_ffmpeg_process, name="hls_ffmpeg")
                    close_proc(save_ffmpeg_process, name="save_ffmpeg")

                    print(f"[Save] 最终视频已保存到: {final_video_path}", flush=True)

                    update_task_status(
                        task_id,
                        status="finished",
                        stage="finished",
                        message="生成完成",
                        total_chunks=iter_total_num,
                        generated_chunks=iter_total_num,
                        is_done=True,
                        stream_ready=True,
                        error=None,
                        final_video_path=final_video_path,
                    )
                else:
                    # 文件模式：合并所有视频块并保存为 MP4
                    update_task_status(
                        task_id,
                        status="running",
                        stage="finalizing",
                        message="正在合并视频并添加音频",
                        is_done=False,
                    )
                    if all_video_chunks:
                        from diffusers.utils import export_to_video
                        import subprocess as sp
                        # 合并所有 chunk（时间维度拼接）
                        full_video = torch.cat(all_video_chunks, dim=2)   # [1, 3, T, H, W]
                        # 转换为 numpy 数组，值域 [0,1]，形状 [T, H, W, 3]
                        video_np = ((full_video.squeeze(0).permute(1, 2, 3, 0).float() + 1.0) / 2)\
                            .clamp(0, 1)\
                            .cpu()\
                            .numpy()
                        # 保存临时无声视频
                        temp_video_path = os.path.join(task_hls_dir, "temp_no_audio.mp4")
                        export_to_video(video_np, temp_video_path, fps=fps)
                        # 添加音频到最终视频
                        final_video_path = os.path.join(self.video_save_root, f"{task_id}.mp4")
                        # 使用 ffmpeg 合并音频（需要 add_audio_to_video 函数，可从 generate.py 复制）
                        add_audio_to_video(temp_video_path, audio_path, final_video_path)
                        # 清理临时文件
                        os.remove(temp_video_path)
                        # 更新任务状态
                        update_task_status(
                            task_id,
                            status="finished",
                            stage="finished",
                            message="生成完成",
                            total_chunks=iter_total_num,
                            generated_chunks=iter_total_num,
                            is_done=True,
                            error=None,
                            final_video_path=final_video_path,
                        )
                        print(f"[Save] 最终视频已保存到: {final_video_path}", flush=True)
                    else:
                        # 没有任何视频块（异常情况）
                        update_task_status(
                            task_id,
                            status="failed",
                            stage="failed",
                            message="没有生成任何视频帧",
                            is_done=True,
                            error="No video chunks generated",
                        )

        except Exception as e:
            print(f"[Generate] 生成失败: {e}", flush=True)

            if self.rank == 0:
                try:
                    close_proc(hls_ffmpeg_process, name="hls_ffmpeg")
                    close_proc(save_ffmpeg_process, name="save_ffmpeg")
                except Exception:
                    pass

                update_task_status(
                    task_id,
                    status="failed",
                    stage="failed",
                    message=f"生成失败: {e}",
                    is_done=True,
                    error=str(e),
                )
                streaming_active = False
            raise

        finally:
            if self.rank == 0:
                streaming_active = False


# ================= 4. Flask 路由 (与前端对接) =================
def control_loop_rank0():
    global streaming_active

    while True:
        try:
            params = task_queue.get(timeout=1.0)
        except queue.Empty:
            params = None

        if engine.use_dist:
            payload = [params]
            dist.broadcast_object_list(payload, src=0, group=engine.control_pg)

        if params is None:
            continue

        try:
            update_task_status(
                params['task_id'],
                status="running",
                stage="starting",
                message="任务开始执行"
            )
            engine.generate_and_push(params)
        finally:
            streaming_active = False


def control_loop_rank_other():
    if not engine.use_dist:
        return
    while True:
        payload = [None]
        dist.broadcast_object_list(payload, src=0, group=engine.control_pg)
        params = payload[0]

        if params is None:
            continue

        engine.generate_and_push(params)
        torch.cuda.empty_cache()
        gc.collect()


@app.route('/')
def index():
    return render_template('index.html', stream_resolution=engine.args.size.replace('*', 'x'))


@app.route('/start_stream', methods=['POST'])
def start_stream():
    global streaming_active

    if streaming_active:
        return jsonify({"status": "error", "message": "GPU 任务繁忙，请稍后再试"}), 429

    task_id = request.form.get('task_id')
    main_prompt = (request.form.get('main_prompt') or '').strip()
    prompt_json = request.form.get('prompt_json') or '[]'
    fps = request.form.get('fps')
    prompt_list = json.loads(prompt_json)
    output_mode = request.form.get('output_mode', 'stream')  # 默认流式
    if output_mode not in ('stream', 'file'):
        output_mode = 'stream'  # 非法值回退

    stream_with_audio = str(request.form.get('stream_with_audio', 'false')).lower() in ('1', 'true', 'yes', 'on')
    img_file = request.files.get('img_file')
    audio_file = request.files.get('audio_file')
    if not task_id:
        return jsonify({"status": "error", "message": "缺少 task_id"}), 400
    if not img_file or not audio_file:
        return jsonify({"status": "error", "message": "缺少图片或音频文件"}), 400
    if not fps:
        return jsonify({"status": "error", "message": "缺少 fps"}), 400
    task_upload_dir = os.path.join(UPLOAD_ROOT, task_id)
    os.makedirs(task_upload_dir, exist_ok=True)
    img_path = os.path.join(task_upload_dir, "input.png")
    audio_path = os.path.join(task_upload_dir, "input.wav")
    img_file.save(img_path)
    audio_file.save(audio_path)
    params = {
        'task_id': task_id,
        'prompt_list': prompt_list,
        'main_prompt': main_prompt,
        'fps': int(fps),
        'img_path': img_path,
        'audio_path': audio_path,
        'stream_with_audio': stream_with_audio,
        'output_mode': output_mode,   # 新增
    }

    update_task_status(
        task_id,
        status="queued",
        stage="queued",
        message="任务已入队，等待执行",
        total_chunks=None,
        generated_chunks=0,
        is_done=False,
        stream_ready=False,
        error=None,
        stream_with_audio=stream_with_audio, )

    streaming_active = True
    task_queue.put(params)
    return jsonify({
        "status": "success",
        "task_id": task_id,
        "stream_with_audio": stream_with_audio
    })


@app.route('/stream/<task_id>/<path:filename>')
def serve_hls(task_id, filename):
    return send_from_directory(os.path.join(HLS_ROOT, task_id), filename)


@app.route('/download/<file_name>')
def download_video(file_name):
    # video_path = os.path.join(engine.video_save_root, f"{task_id}.mp4")
    video_path = os.path.join(engine.video_save_root, file_name)
    if not os.path.exists(video_path):
        abort(404, description="Video file not found")

    delete = request.args.get('delete', '0')
    if delete and delete != '0':
        @after_this_request
        def remove_file(response):
            try:
                print(u'删除文件：{}……'.format(video_path))
                os.remove(video_path)
            except Exception as e:
                app.logger.error(f"Failed to delete file {video_path}: {e}")
            return response

    return send_file(video_path, as_attachment=True, mimetype='video/mp4')


@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    data = get_task_status(task_id)
    if data is None:
        return jsonify({
            "status": "not_found",
            "message": "task_id 不存在"
        }), 404
    return jsonify(data)


# ================= 5. 分布式启动 =================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--wav2vec_dir", type=str, required=True)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--size", type=str, default="720*416",
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.")
    parser.add_argument("--video_save_path", type=str, default="./generated_videos",
                        help="Directory to save final generated videos.")
    parser.add_argument(
        "--fp8_kv_cache",
        action="store_true",
        default=False,
        help="Whether to store kv cache in FP8 and dequantize to BF16 on use.")
    parser.add_argument(
        "--block_offload",
        action="store_true",
        default=False,
        help="Whether to offload WanModel blocks to CPU between block forwards.")
    # parser.add_argument(
    #     "--output_mode",
    #     type=str,
    #     default="stream",
    #     choices=["stream", "file"],
    #     help="Output mode: 'stream' for real-time HLS streaming, 'file' for offline MP4 generation"
    # )
    args = parser.parse_args()

    try:
        engine = DistributedVideoEngine(args)
        if engine.rank == 0:
            threading.Thread(target=control_loop_rank0, daemon=True).start()

            ip = get_local_ip()
            print(f"\n🚀 LiveAct 服务启动!")
            print(f"访问地址: http://{ip}:{args.port}\n")
            app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
        else:
            print(f"节点 Rank {engine.rank} 等待指令...")
            control_loop_rank_other()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

