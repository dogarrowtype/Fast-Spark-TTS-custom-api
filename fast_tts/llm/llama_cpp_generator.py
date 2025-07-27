# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:52
# Author    :Hui Huang
import os
from typing import AsyncIterator, Optional, List
from ..logger import get_logger
from .base_llm import BaseLLM, GenerationResponse

def _check_cuda_availability():
    """Check if CUDA is available for llama-cpp-python"""
    try:
        from llama_cpp import _load_shared_library
        # Check if llama-cpp was compiled with CUDA support
        #return hasattr(llama_cpp.llama_cpp, 'GGML_USE_CUBLAS') or hasattr(llama_cpp.llama_cpp, 'GGML_USE_CUDA')
        lib = _load_shared_library('llama')
        return bool(lib.llama_supports_gpu_offload())
    except (ImportError, AttributeError):
        return False

def _verify_cuda_usage(model, device: str, n_gpu_layers: int):
    """Verify that the model is actually using CUDA when requested"""
    if not device.startswith('cuda'):
        return  # Not using CUDA, no need to verify
    
    logger = get_logger()
    
    # Check if CUDA support was compiled in
    if not _check_cuda_availability():
        logger.warning("CUDA device was requested but llama-cpp-python was not compiled with CUDA support. "
                      "The model will fall back to CPU. Please install llama-cpp-python with CUDA support.")
        return
    
    # Try to get model metadata to verify GPU usage
    try:
        # Check if the model has GPU layers loaded
        if hasattr(model, 'n_gpu_layers'):
            actual_gpu_layers = model.n_gpu_layers
            if actual_gpu_layers == 0:
                logger.warning(f"CUDA device '{device}' was requested with {n_gpu_layers} GPU layers, "
                             f"but the model is using 0 GPU layers. Check CUDA installation and GPU memory.")
            else:
                logger.info(f"Successfully verified CUDA usage: {actual_gpu_layers} GPU layers loaded on device '{device}'")
        else:
            # Alternative check - try to access CUDA context if available
            try:
                import llama_cpp
                # If we can access the model's context and it's using CUDA, this should work
                if hasattr(model, 'ctx') and model.ctx:
                    logger.info(f"CUDA device '{device}' appears to be in use (model context initialized)")
                else:
                    logger.warning(f"CUDA device '{device}' was requested but model context verification failed")
            except Exception as e:
                logger.warning(f"Could not verify CUDA usage for device '{device}': {e}")
                
    except Exception as e:
        logger.warning(f"Could not verify CUDA usage for device '{device}': {e}")

logger = get_logger()

__all__ = ["LlamaCppGenerator"]


class LlamaCppGenerator(BaseLLM):
    def __init__(
            self,
            model_path: str,
            max_length: int = 32768,
            device: str = "cpu",
            stop_tokens: Optional[list[str]] = None,
            stop_token_ids: Optional[List[int]] = None,
            **kwargs
    ):
        from llama_cpp import Llama

        model_files = []
        for filename in os.listdir(model_path):
            if filename.endswith(".gguf"):
                model_files.append(filename)
        if len(model_files) == 0:
            logger.error("No gguf file found in the model directory")
            raise ValueError("No gguf file found in the model directory")
        else:
            if len(model_files) > 1:
                logger.warning(
                    f"Multiple gguf files found in the model directory, using the first one: {model_files[0]}")
            model_file = os.path.join(model_path, model_files[0])

        # Improved CUDA handling for llama-cpp
        if device == 'cpu':
            n_gpu_layers = 0
            logger.info("Using CPU-only mode for llama-cpp")
            runtime_kwargs = dict(
                model_path=model_file,
                n_ctx=max_length,
                n_gpu_layers=n_gpu_layers,
                verbose=False,  # Suppress debug output
                **kwargs
            )
        elif device.startswith('cuda'):
            n_gpu_layers = -1  # Use all GPU layers
            # Extract CUDA device ID if specified (e.g., cuda:1)
            if ':' in device:
                gpu_id = int(device.split(':')[1])
                logger.info(f"Using CUDA device {gpu_id} with all GPU layers for llama-cpp")
                runtime_kwargs = dict(
                    model_path=model_file,
                    n_ctx=max_length,
                    n_gpu_layers=n_gpu_layers,
                    main_gpu=gpu_id,
                    verbose=False,  # Suppress debug output
                    **kwargs
                )
            else:
                logger.info("Using CUDA with all GPU layers for llama-cpp")
                runtime_kwargs = dict(
                    model_path=model_file,
                    n_ctx=max_length,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,  # Suppress debug output
                    **kwargs
                )
        else:
            # Default fallback
            n_gpu_layers = -1
            logger.info("Using default GPU configuration for llama-cpp")
            runtime_kwargs = dict(
                model_path=model_file,
                n_ctx=max_length,
                n_gpu_layers=n_gpu_layers,
                verbose=False,  # Suppress debug output
                **kwargs
            )
        self.model = Llama(
            **runtime_kwargs
        )
        
        # Verify CUDA usage if a CUDA device was requested
        _verify_cuda_usage(self.model, device, n_gpu_layers)
        # 不使用llama cpp 的 tokenizer
        super(LlamaCppGenerator, self).__init__(
            tokenizer=model_path,
            max_length=max_length,
            stop_tokens=stop_tokens,
            stop_token_ids=stop_token_ids,
        )

    async def _generate(
            self,
            prompt_ids: list[int],
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1.0,
            skip_special_tokens: bool = True,
            **kwargs
    ) -> GenerationResponse:
        completion_tokens = []
        for token in self.model.generate(
                prompt_ids,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                repeat_penalty=repetition_penalty,
                **kwargs
        ):
            if token in self.stop_token_ids:
                break

            if len(completion_tokens) + len(prompt_ids) > self.max_length:
                break
            completion_tokens.append(token)

        # Decode the generated tokens into text
        output = self.tokenizer.decode(
            completion_tokens,
            skip_special_tokens=skip_special_tokens
        )
        return GenerationResponse(text=output, token_ids=completion_tokens)

    async def _stream_generate(
            self,
            prompt_ids: list[int],
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1.0,
            skip_special_tokens: bool = True,
            **kwargs
    ) -> AsyncIterator[GenerationResponse]:
        completion_tokens = []
        previous_texts = ""
        for token in self.model.generate(
                prompt_ids,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                repeat_penalty=repetition_penalty,
                **kwargs
        ):
            if token in self.stop_token_ids:
                break
            if len(completion_tokens) + len(prompt_ids) > self.max_length:
                break
            completion_tokens.append(token)

            text = self.tokenizer.decode(completion_tokens, skip_special_tokens=skip_special_tokens)

            delta_text = text[len(previous_texts):]
            previous_texts = text

            yield GenerationResponse(
                text=delta_text,
                token_ids=[token],
            )
