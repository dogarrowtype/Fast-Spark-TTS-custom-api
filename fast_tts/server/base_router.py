# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/7 16:14
# Author  : Hui Huang
import base64
import io

import httpx
import numpy as np
from fastapi import HTTPException, Request, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from .protocol import TTSRequest, CloneRequest, SpeakRequest, MultiSpeakRequest
from ..engine import AutoEngine
from ..logger import get_logger
import soundfile as sf

logger = get_logger()

base_router = APIRouter(
    tags=["Fast-TTS"],
    responses={404: {"description": "Not found"}},
)


async def process_audio_buffer(audio: np.ndarray, sr: int):
    audio_io = io.BytesIO()
    sf.write(audio_io, audio, sr, format="WAV", subtype="PCM_16")
    audio_io.seek(0)
    return audio_io


# TTS 合成接口：接收 JSON 请求，返回合成语音（wav 格式）
@base_router.post("/generate_voice")
async def generate_voice(req: TTSRequest, raw_request: Request):
    engine: AutoEngine = raw_request.app.state.engine
    if engine.engine_name == 'orpheus':
        logger.error("OrpheusTTS 暂不支持语音合成.")
        raise HTTPException(status_code=500, detail="OrpheusTTS 暂不支持该功能.")
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
                    length_threshold=req.length_threshold,
                    window_size=req.window_size
            ):
                audio_bytes = chunk.tobytes()
                yield audio_bytes

        return StreamingResponse(generate_audio_stream(), media_type="audio/wav")
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
                length_threshold=req.length_threshold,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning(f"TTS 合成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        audio_io = await process_audio_buffer(audio, sr=engine.SAMPLE_RATE)
        return StreamingResponse(audio_io, media_type="audio/wav")


async def get_audio_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法从指定 URL 下载参考音频")
        return response.content


@base_router.get("/sample_rate")
async def sample_rate(raw_request: Request):
    sr = raw_request.app.state.engine.SAMPLE_RATE
    return JSONResponse(
        content={
            "success": True,
            "sample_rate": sr
        })


# 克隆语音接口：接收 multipart/form-data，上传参考音频和其它表单参数
@base_router.post("/clone_voice")
async def clone_voice(
        req: CloneRequest, raw_request: Request
):
    engine: AutoEngine = raw_request.app.state.engine
    if engine.engine_name == 'orpheus':
        logger.error("OrpheusTTS 暂不支持语音克隆.")
        raise HTTPException(status_code=500, detail="OrpheusTTS 暂不支持该功能.")

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
                    length_threshold=req.length_threshold,
                    window_size=req.window_size
            ):
                out_bytes = chunk.tobytes()
                yield out_bytes

        return StreamingResponse(generate_audio_stream(), media_type="audio/wav")
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
                length_threshold=req.length_threshold,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning("生成克隆语音失败：" + str(e))
            raise HTTPException(status_code=500, detail="生成克隆语音失败：" + str(e))

        audio_io = await process_audio_buffer(audio, sr=engine.SAMPLE_RATE)
        return StreamingResponse(audio_io, media_type="audio/wav")


@base_router.get("/audio_roles")
async def audio_roles(raw_request: Request):
    roles = raw_request.app.state.engine.list_roles()
    return JSONResponse(
        content={
            "success": True,
            "roles": roles
        })


@base_router.post("/speak")
async def speak(req: SpeakRequest, raw_request: Request):
    engine: AutoEngine = raw_request.app.state.engine
    if req.name not in engine.list_roles():
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
                    length_threshold=req.length_threshold,
                    window_size=req.window_size
            ):
                out_bytes = chunk.tobytes()
                yield out_bytes

        return StreamingResponse(generate_audio_stream(), media_type="audio/wav")
    else:
        try:
            audio = await engine.speak_async(
                name=req.name,
                text=req.text,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                max_tokens=req.max_tokens,
                length_threshold=req.length_threshold,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning(f"TTS 合成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        audio_io = await process_audio_buffer(audio, sr=engine.SAMPLE_RATE)
        return StreamingResponse(audio_io, media_type="audio/wav")


@base_router.post("/multi_speak")
async def multi_speak(req: MultiSpeakRequest, raw_request: Request):
    engine: AutoEngine = raw_request.app.state.engine

    if req.stream:
        async def generate_audio_stream():
            async for chunk in engine.multi_speak_stream_async(
                    text=req.text,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    max_tokens=req.max_tokens,
                    length_threshold=req.length_threshold,
                    window_size=req.window_size
            ):
                out_bytes = chunk.tobytes()
                yield out_bytes

        return StreamingResponse(generate_audio_stream(), media_type="audio/wav")
    else:
        try:
            audio = await engine.multi_speak_async(
                text=req.text,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                max_tokens=req.max_tokens,
                length_threshold=req.length_threshold,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning(f"TTS 合成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        audio_io = await process_audio_buffer(audio, sr=engine.SAMPLE_RATE)
        return StreamingResponse(audio_io, media_type="audio/wav")
