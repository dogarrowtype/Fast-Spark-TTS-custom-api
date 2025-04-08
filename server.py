# -*- coding: utf-8 -*-
# Time      :2025/3/15 11:37
# Author    :Hui Huang
import argparse
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from starlette.middleware.cors import CORSMiddleware

from fast_tts import (
    AutoEngine,
    get_logger,
    setup_logging
)
from fast_tts.server.base_router import base_router
from fast_tts.server.openai_router import openai_router

logger = get_logger()


async def load_roles(async_engine: AutoEngine, role_dir: Optional[str] = None):
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


async def warmup_engine(async_engine: AutoEngine):
    logger.info("Warming up...")
    if async_engine.engine_name == 'spark':
        await async_engine.generate_voice_async(
            text="测试音频",
            max_tokens=128
        )
    elif async_engine.engine_name == 'orpheus':
        await async_engine.speak_async(
            name='tara',
            text="test audio.",
            max_tokens=128
        )
    logger.info("Warmup complete.")


def build_app(args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 使用解析到的参数初始化全局 TTS 引擎
        engine = AutoEngine(
            model_path=args.model_path,
            snac_path=args.snac_path,
            max_length=args.max_length,
            llm_device=args.llm_device,
            tokenizer_device=args.tokenizer_device,
            detokenizer_device=args.detokenizer_device,
            backend=args.backend,
            wav2vec_attn_implementation=args.wav2vec_attn_implementation,
            llm_attn_implementation=args.llm_attn_implementation,
            llm_gpu_memory_utilization=args.llm_gpu_memory_utilization,
            torch_dtype=args.torch_dtype,
            batch_size=args.batch_size,
            llm_batch_size=args.llm_batch_size,
            wait_timeout=args.wait_timeout,
            cache_implementation=args.cache_implementation,
            seed=args.seed
        )
        if engine.engine_name == 'spark':
            await load_roles(engine, args.role_dir)
        await warmup_engine(engine)
        # 将 engine 保存到 app.state 中，方便路由中使用
        app.state.engine = engine
        yield

    app = FastAPI(lifespan=lifespan)

    app.include_router(base_router)
    app.include_router(openai_router, prefix="/v1")

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
    parser = argparse.ArgumentParser(description="FastTTS 后端")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")

    parser.add_argument("--backend", type=str, required=True,
                        choices=["llama-cpp", "vllm", "sglang", "torch", "mlx-lm"],
                        help="引擎类型，如 llama-cpp、vllm、sglang、mlx-lm 或 torch")
    parser.add_argument("--snac_path", type=str, default=None,
                        help="OrpheusTTS 的snac模块地址")
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
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=1, help="音频处理组件单批次处理的最大请求数。")
    parser.add_argument("--llm_batch_size", type=int, default=256, help="LLM模块单批次处理的最大请求数。")
    parser.add_argument("--wait_timeout", type=float, default=0.01, help="动态批处理请求超时阈值，单位为秒。")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="服务监听端口")
    args = parser.parse_args()

    setup_logging()

    logger.info("启动 FastTTS 服务")
    logger.info(f"Config: {args}")
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
