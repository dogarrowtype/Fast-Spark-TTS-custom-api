# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/7 16:09
# Author  : Hui Huang
import time
from typing import Literal, Optional, List

from pydantic import BaseModel, Field


# 定义 TTS 合成请求体（JSON 格式）
class TTSRequest(BaseModel):
    text: str
    gender: Literal["female", "male"] = "female"
    pitch: Literal["very_low", "low", "moderate", "high", "very_high"] = "moderate"
    speed: Literal["very_low", "low", "moderate", "high", "very_high"] = "moderate"
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 4096
    length_threshold: int = 50
    window_size: int = 50
    stream: bool = False


# 定义支持多种方式传入参考音频的请求协议
class CloneRequest(BaseModel):
    text: str
    # reference_audio 字段既可以是一个 URL，也可以是 base64 编码的音频数据
    reference_audio: str
    reference_text: Optional[str] = None
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 4096
    length_threshold: int = 50
    window_size: int = 50
    stream: bool = False


# 定义角色语音合成请求体
class SpeakRequest(BaseModel):
    name: str
    text: str
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 4096
    length_threshold: int = 50
    window_size: int = 50
    stream: bool = False


# 定义多角色语音合成请求体
class MultiSpeakRequest(BaseModel):
    text: str
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 4096
    length_threshold: int = 50
    window_size: int = 50
    stream: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "FastTTS"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


# copy from https://github.com/remsky/Kokoro-FastAPI/blob/master/api/src/routers/openai_compatible.py
class OpenAISpeechRequest(BaseModel):
    model: str = Field(
        default=None,
        description="The model to use for generation.",
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default=None,
        description="The voice to use for generation. Can be a base voice or a combined voice name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, flac, wav, pcm. PCM format returns raw 16-bit samples without headers. AAC is not currently supported.",
    )
    stream: bool = Field(
        default=True,
        description="If true, audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
