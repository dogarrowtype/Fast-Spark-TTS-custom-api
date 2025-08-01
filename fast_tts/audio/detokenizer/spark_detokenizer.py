# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:34
# Author    :Hui Huang
import os
from typing import Literal

import torch
from ..base_model import SparkBaseModel
from ..batch_processor import AsyncBatchEngine
from ...modules.encoder_decoder.feat_decoder import Decoder
from ...modules.encoder_decoder.wave_generator import WaveGenerator
from ...modules.speaker.speaker_encoder import SpeakerEncoder
from ...modules.vq.factorized_vector_quantize import FactorizedVectorQuantize
from ...logger import get_logger

logger = get_logger()

__all__ = ["SparkDeTokenizer"]


class SparkDeTokenizerModel(SparkBaseModel):
    def __init__(self, config):
        super().__init__()

        self.quantizer = FactorizedVectorQuantize(**config["quantizer"])
        self.prenet = Decoder(**config["prenet"])
        self.decoder = WaveGenerator(**config["decoder"])
        self.speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

    @torch.no_grad()
    def forward(
            self,
            semantic_tokens: torch.Tensor,
            global_tokens: torch.Tensor
    ) -> torch.Tensor:
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)
        return wav_recon.detach()


class SparkDeTokenizer:
    def __init__(
            self,
            model_path: str,
            device: Literal["cpu", "cuda", "mps"] | str = "cpu",
            batch_size: int = 32,
            wait_timeout: float = 0.01):
        # Validate and set device with better error handling
        try:
            self.device = torch.device(device)
            logger.info(f"SparkDeTokenizer initialized with device: {self.device}")
        except Exception as e:
            logger.error(f"Invalid device '{device}' for SparkDeTokenizer: {e}, falling back to CPU")
            self.device = torch.device("cpu")
            
        self.model = SparkDeTokenizerModel.from_pretrained(
            os.path.join(model_path, "BiCodec")
        ).to(self.device)
        
        # Enable device-specific optimizations
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Applying CUDA optimizations for SparkDeTokenizer")
        else:
            logger.info(f"Using {self.device.type} mode for SparkDeTokenizer")

        self._batch_processor = AsyncBatchEngine(
            processing_function=self.batch_detokenize_async,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )

    @torch.no_grad()
    def detokenize(
            self,
            semantic_tokens: torch.Tensor,
            global_tokens: torch.Tensor
    ) -> torch.Tensor:
        output = self.model(
            semantic_tokens.to(self.device),
            global_tokens.to(self.device)
        )
        return output

    async def batch_detokenize_async(self, requests: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        semantic_tokens, global_tokens = [], []
        lengths = []
        for request in requests:
            semantic_tokens.append(request["semantic_tokens"])
            global_tokens.append(request["global_tokens"])
            lengths.append(len(request['semantic_tokens']))
        # Concatenate tokens for batch processing
        global_tokens = torch.stack(global_tokens, dim=0)
        semantic_tokens = torch.nn.utils.rnn.pad_sequence(
            semantic_tokens, batch_first=True, padding_value=0
        )

        audios = self.detokenize(
            semantic_tokens=semantic_tokens,
            global_tokens=global_tokens
        ).detach().cpu()
        # Prepare responses
        responses = []
        for i in range(len(requests)):
            audio = audios[i, :, :(lengths[i] * 320)]  # 大概一个token对应audio长度320
            responses.append({
                "audio": audio,
            })

        # Enhanced CUDA memory management
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return responses

    async def detokenize_async(self, request: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = await self._batch_processor.add_request(
            single_input=request
        )
        return output.get("feature")
