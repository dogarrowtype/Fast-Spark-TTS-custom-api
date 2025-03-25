# -*- coding: utf-8 -*-
# Time      :2025/3/15 11:37
# Author    :Hui Huang
import argparse
import base64
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import io
import uvicorn
import soundfile as sf
from starlette.middleware.cors import CORSMiddleware

from fast_sparktts import (
    AsyncFastSparkTTS,
    get_logger,
    setup_logging
)

logger = get_logger()

router = APIRouter()


# 定义 TTS 合成请求体（JSON 格式）
class TTSRequest(BaseModel):
    text: str
    gender: Literal["female", "male"] = "female"
    pitch: Literal["very_low", "low", "moderate", "high", "very_high"] = "moderate"
    speed: Literal["very_low", "low", "moderate", "high", "very_high"] = "moderate"
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 4096
    split: bool = False
    window_size: int = 100
    stream: bool = False
    audio_chunk_duration: float = 1.0
    max_audio_chunk_duration: float = 8.0
    audio_chunk_size_scale_factor: float = 2.0
    audio_chunk_overlap_duration: float = 0.1


# 定义支持多种方式传入参考音频的请求协议
class CloneRequest(BaseModel):
    text: str
    # reference_audio 字段既可以是一个 URL，也可以是 base64 编码的音频数据
    reference_audio: str
    reference_text: Optional[str] = None
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 4096
    split: bool = False
    window_size: int = 100
    stream: bool = False
    audio_chunk_duration: float = 1.0
    max_audio_chunk_duration: float = 8.0
    audio_chunk_size_scale_factor: float = 2.0
    audio_chunk_overlap_duration: float = 0.1


# 定义角色语音合成请求体
class SpeakRequest(BaseModel):
    name: str
    text: str
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 4096
    split: bool = False
    window_size: int = 100
    stream: bool = False
    audio_chunk_duration: float = 1.0
    max_audio_chunk_duration: float = 8.0
    audio_chunk_size_scale_factor: float = 2.0
    audio_chunk_overlap_duration: float = 0.1


async def load_roles(async_engine: AsyncFastSparkTTS, role_dir: Optional[str] = None):
    # 加载已有的角色音频
    if role_dir is not None and os.path.exists(role_dir):
        logger.info(f"加载角色库：{role_dir}")
        role_list = os.listdir(role_dir)
        exist_roles = []
        for role in role_list:
            if role in exist_roles:
                logger.warning(f"{role} 角色已存在")
                continue
            role_audio = os.path.join(role_dir, role, "reference_audio.wav")

            try:
                role_text = open(os.path.join(role_dir, role, "reference_text.txt"), "r", encoding='utf8').read()
                role_text = role_text.strip()
            except Exception as e:
                role_text = None

            if role_text == "":
                role_text = None

            exist_roles.append(role)
            await async_engine.add_speaker(
                name=role,
                audio=role_audio,
                reference_text=role_text,
            )
        logger.info(f"角色库加载完毕，角色有：{'、'.join(exist_roles)}")
    else:
        logger.warning("当前角色目录不存在，将无法使用角色克隆功能！")


async def warmup_engine(async_engine: AsyncFastSparkTTS):
    logger.info("Warming up...")
    await async_engine.generate_voice_async(
        text="测试音频",
        max_tokens=128
    )
    logger.info("Warmup complete.")


async def process_audio_buffer(audio: np.ndarray):
    audio_io = io.BytesIO()
    sf.write(audio_io, audio, 16000, format="WAV", subtype="PCM_16")
    audio_io.seek(0)
    return audio_io


# TTS 合成接口：接收 JSON 请求，返回合成语音（wav 格式）
@router.post("/generate_voice")
async def generate_voice(req: TTSRequest, raw_request: Request):
    engine: AsyncFastSparkTTS = raw_request.app.state.engine
    if req.stream:
        async def generate_audio_stream():
            async for chunk in engine.generate_voice_stream_async(
                    req.text,
                    gender=req.gender,
                    pitch=req.pitch,
                    speed=req.speed,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    max_tokens=req.max_tokens,
                    split=req.split,
                    window_size=req.window_size,
                    audio_chunk_duration=req.audio_chunk_duration,
                    max_audio_chunk_duration=req.max_audio_chunk_duration,
                    audio_chunk_size_scale_factor=req.audio_chunk_size_scale_factor,
                    audio_chunk_overlap_duration=req.audio_chunk_overlap_duration,
            ):
                audio_bytes = (chunk * (2 ** 15)).astype(np.int16).tobytes()
                yield audio_bytes

        return StreamingResponse(generate_audio_stream(), media_type="audio/pcm")
    else:
        try:
            audio = await engine.generate_voice_async(
                req.text,
                gender=req.gender,
                pitch=req.pitch,
                speed=req.speed,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                max_tokens=req.max_tokens,
                split=req.split,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning(f"TTS 合成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        audio_io = await process_audio_buffer(audio)
        return StreamingResponse(audio_io, media_type="audio/wav")


async def get_audio_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法从指定 URL 下载参考音频")
        return response.content


# 克隆语音接口：接收 multipart/form-data，上传参考音频和其它表单参数
@router.post("/clone_voice")
async def clone_voice(
        req: CloneRequest, raw_request: Request
):
    engine: AsyncFastSparkTTS = raw_request.app.state.engine

    # 根据 reference_audio 内容判断读取方式
    if req.reference_audio.startswith("http://") or req.reference_audio.startswith("https://"):
        audio_bytes = await get_audio_bytes_from_url(req.reference_audio)
    else:
        try:
            audio_bytes = base64.b64decode(req.reference_audio)
        except Exception as e:
            logger.warning("无效的 base64 音频数据: " + str(e))
            raise HTTPException(status_code=400, detail="无效的 base64 音频数据: " + str(e))
    # 利用 BytesIO 包装字节数据，然后使用 soundfile 读取为 numpy 数组
    try:
        bytes_io = io.BytesIO(audio_bytes)
    except Exception as e:
        logger.warning("读取参考音频失败: " + str(e))
        raise HTTPException(status_code=400, detail="读取参考音频失败: " + str(e))

    if req.stream:
        async def generate_audio_stream():
            async for chunk in engine.clone_voice_stream_async(
                    text=req.text,
                    reference_audio=bytes_io,
                    reference_text=req.reference_text,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    max_tokens=req.max_tokens,
                    split=req.split,
                    window_size=req.window_size,
                    audio_chunk_duration=req.audio_chunk_duration,
                    max_audio_chunk_duration=req.max_audio_chunk_duration,
                    audio_chunk_size_scale_factor=req.audio_chunk_size_scale_factor,
                    audio_chunk_overlap_duration=req.audio_chunk_overlap_duration,
            ):
                out_bytes = (chunk * (2 ** 15)).astype(np.int16).tobytes()
                yield out_bytes

        return StreamingResponse(generate_audio_stream(), media_type="audio/pcm")
    else:
        try:
            audio = await engine.clone_voice_async(
                text=req.text,
                reference_audio=bytes_io,
                reference_text=req.reference_text,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                max_tokens=req.max_tokens,
                split=req.split,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning("生成克隆语音失败：" + str(e))
            raise HTTPException(status_code=500, detail="生成克隆语音失败：" + str(e))

        audio_io = await process_audio_buffer(audio)
        return StreamingResponse(audio_io, media_type="audio/wav")


@router.get("/audio_roles")
async def audio_roles(raw_request: Request):
    roles = raw_request.app.state.engine.list_roles()
    return JSONResponse(
        content={
            "success": True,
            "roles": roles
        })


@router.post("/speak")
async def speak(req: SpeakRequest, raw_request: Request):
    engine: AsyncFastSparkTTS = raw_request.app.state.engine
    if req.name not in engine.speakers:
        err_msg = f"{req.name} 不在已有的角色列表中。"
        logger.warning(err_msg)
        raise HTTPException(status_code=500, detail=err_msg)

    if req.stream:
        async def generate_audio_stream():
            async for chunk in engine.speak_stream_async(
                    name=req.name,
                    text=req.text,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    max_tokens=req.max_tokens,
                    split=req.split,
                    window_size=req.window_size,
                    audio_chunk_duration=req.audio_chunk_duration,
                    max_audio_chunk_duration=req.max_audio_chunk_duration,
                    audio_chunk_size_scale_factor=req.audio_chunk_size_scale_factor,
                    audio_chunk_overlap_duration=req.audio_chunk_overlap_duration,
            ):
                out_bytes = (chunk * (2 ** 15)).astype(np.int16).tobytes()
                yield out_bytes

        return StreamingResponse(generate_audio_stream(), media_type="audio/pcm")
    else:
        try:
            audio = await engine.speak_async(
                name=req.name,
                text=req.text,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                max_tokens=req.max_tokens,
                split=req.split,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning(f"TTS 合成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        audio_io = await process_audio_buffer(audio)
        return StreamingResponse(audio_io, media_type="audio/wav")


def build_app(args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 使用解析到的参数初始化全局 TTS 引擎
        engine = AsyncFastSparkTTS(
            model_path=args.model_path,
            max_length=args.max_length,
            llm_device=args.llm_device,
            tokenizer_device=args.tokenizer_device,
            detokenizer_device=args.detokenizer_device,
            engine=args.engine,
            wav2vec_attn_implementation=args.wav2vec_attn_implementation,
            llm_attn_implementation=args.llm_attn_implementation,
            llm_gpu_memory_utilization=args.llm_gpu_memory_utilization,
            torch_dtype=args.torch_dtype,
            batch_size=args.batch_size,
            wait_timeout=args.wait_timeout,
            cache_implementation=args.cache_implementation,
            seed=args.seed
        )
        await load_roles(engine, args.role_dir)
        await warmup_engine(engine)
        # 将 engine 保存到 app.state 中，方便路由中使用
        app.state.engine = engine
        yield

    app = FastAPI(lifespan=lifespan)

    app.include_router(router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if args.api_key is not None:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if request.method == "OPTIONS":
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + args.api_key:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    return app


if __name__ == '__main__':
    # 使用 argparse 获取启动参数
    parser = argparse.ArgumentParser(description="FastSparkTTS 后端")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--engine", type=str, required=True,
                        choices=["llama-cpp", "vllm", "sglang", "torch"],
                        help="引擎类型，如 llama-cpp、vllm、sglang 或 torch")
    parser.add_argument("--llm_device", type=str, default="auto",
                        help="llm 设备，例如 cpu 或 cuda:0")
    parser.add_argument("--tokenizer_device", type=str, default="auto",
                        help="audio tokenizer 设备")
    parser.add_argument("--detokenizer_device", type=str, default="auto",
                        help="audio detokenizer 设备")
    parser.add_argument("--wav2vec_attn_implementation", type=str, default="eager",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="wav2vec 的 attn_implementation 方式")
    parser.add_argument("--llm_attn_implementation", type=str, default="eager",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="torch generator 的 attn_implementation 方式")
    parser.add_argument("--max_length", type=int, default=32768,
                        help="最大生成长度")
    parser.add_argument("--llm_gpu_memory_utilization", type=float, default=0.6,
                        help="vllm和sglang暂用显存比例，单卡可降低该参数")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                        choices=['float16', "bfloat16", 'float32', 'auto'],
                        help="torch generator中llm使用的dtype。")
    parser.add_argument("--cache_implementation", type=str, default=None,
                        help='将在“generate”中实例化的缓存类的名称，用于更快地解码. 可能设置的值有：static、offloaded_static、sliding_window、hybrid、mamba、quantized。'
                        )
    parser.add_argument("--role_dir", type=str, default="data/roles",
                        help="存放已有角色信息的目录")
    parser.add_argument("--api_key", type=str, default=None,
                        help="设置接口访问限制：Api key")
    parser.add_argument("--seed", type=int, default=8000, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=32, help="音频处理组件单批次处理的最大请求数。")
    parser.add_argument("--wait_timeout", type=float, default=0.01, help="动态批处理请求超时阈值，单位为秒。")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="服务监听端口")
    args = parser.parse_args()

    setup_logging()

    logger.info("启动 FastAPI TTS 服务")
    logger.info(f"Config: {args}")
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
