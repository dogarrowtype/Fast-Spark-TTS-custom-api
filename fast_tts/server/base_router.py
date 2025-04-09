# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/7 16:14
# Author  : Hui Huang
import base64
import io

import httpx
import numpy as np
from fastapi import HTTPException, Request, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse, Response, FileResponse
from .protocol import TTSRequest, CloneRequest, SpeakRequest, MultiSpeakRequest
from .utils.audio_writer import StreamingAudioWriter
from ..engine import AutoEngine
from ..logger import get_logger

logger = get_logger()

base_router = APIRouter(
    tags=["Fast-TTS"],
    responses={404: {"description": "Not found"}},
)


async def generate_audio_stream(generator, data, writer: StreamingAudioWriter, raw_request: Request):
    async for chunk in generator(**data):
        # Check if client is still connected
        is_disconnected = raw_request.is_disconnected
        if callable(is_disconnected):
            is_disconnected = await is_disconnected()
        if is_disconnected:
            logger.info("Client disconnected, stopping audio generation")
            break

        audio = writer.write_chunk(chunk, finalize=False)
        yield audio
    yield writer.write_chunk(finalize=True)


async def generate_audio(audio: np.ndarray, writer: StreamingAudioWriter):
    output = writer.write_chunk(audio, finalize=False)
    final = writer.write_chunk(finalize=True)
    output = output + final
    return output


@base_router.get("/")
async def get_web():
    return FileResponse("templates/index.html")


# TTS 合成接口：接收 JSON 请求，返回合成语音（wav 格式）
@base_router.post("/generate_voice")
async def generate_voice(req: TTSRequest, raw_request: Request):
    engine: AutoEngine = raw_request.app.state.engine
    if engine.engine_name == 'orpheus':
        logger.error("OrpheusTTS 暂不支持语音合成.")
        raise HTTPException(status_code=500, detail="OrpheusTTS 暂不支持该功能.")

    audio_writer = StreamingAudioWriter(req.response_format, sample_rate=engine.SAMPLE_RATE)
    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(req.response_format, f"audio/{req.response_format}")

    if req.stream:
        data = dict(
            text=req.text,
            gender=req.gender,
            pitch=req.pitch,
            speed=req.speed,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
            length_threshold=req.length_threshold,
            window_size=req.window_size
        )
        return StreamingResponse(
            generate_audio_stream(
                engine.generate_voice_stream_async,
                data,
                audio_writer,
                raw_request
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
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
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        audio_io = await generate_audio(audio, writer=audio_writer)
        return Response(
            audio_io,
            media_type=content_type,
            headers=headers,
        )


async def get_audio_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法从指定 URL 下载参考音频")
        return response.content


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

    audio_writer = StreamingAudioWriter(req.response_format, sample_rate=engine.SAMPLE_RATE)
    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(req.response_format, f"audio/{req.response_format}")

    if req.stream:
        data = dict(
            text=req.text,
            reference_audio=bytes_io,
            reference_text=req.reference_text,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
            length_threshold=req.length_threshold,
            window_size=req.window_size
        )
        return StreamingResponse(
            generate_audio_stream(
                engine.clone_voice_stream_async,
                data,
                audio_writer,
                raw_request
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
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
            logger.warning(f"克隆语音失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        audio_io = await generate_audio(audio, writer=audio_writer)
        return Response(
            audio_io,
            media_type=content_type,
            headers=headers,
        )


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

    audio_writer = StreamingAudioWriter(req.response_format, sample_rate=engine.SAMPLE_RATE)
    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(req.response_format, f"audio/{req.response_format}")

    if req.stream:
        data = dict(
            name=req.name,
            text=req.text,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
            length_threshold=req.length_threshold,
            window_size=req.window_size
        )
        return StreamingResponse(
            generate_audio_stream(
                engine.speak_stream_async,
                data,
                audio_writer,
                raw_request
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
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
            logger.warning(f"角色语音合成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        audio_io = await generate_audio(audio, writer=audio_writer)
        return Response(
            audio_io,
            media_type=content_type,
            headers=headers,
        )


@base_router.post("/multi_speak")
async def multi_speak(req: MultiSpeakRequest, raw_request: Request):
    engine: AutoEngine = raw_request.app.state.engine

    audio_writer = StreamingAudioWriter(req.response_format, sample_rate=engine.SAMPLE_RATE)
    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(req.response_format, f"audio/{req.response_format}")

    if req.stream:
        data = dict(
            text=req.text,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
            length_threshold=req.length_threshold,
            window_size=req.window_size
        )
        return StreamingResponse(
            generate_audio_stream(
                engine.multi_speak_stream_async,
                data,
                audio_writer,
                raw_request
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
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
            logger.warning(f"多角色语音合成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        audio_io = await generate_audio(audio, writer=audio_writer)
        return Response(
            audio_io,
            media_type=content_type,
            headers=headers,
        )
