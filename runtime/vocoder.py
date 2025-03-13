# -*- coding: utf-8 -*-
# Time      :2025/3/13 20:22
# Author    :Hui Huang
import os.path
import uuid
from typing import Literal, Optional

import torch

from runtime.batch_processor import AsyncBatchEngine
from sparktts.models.bicodec import BiCodec


class VoCoder:
    def __init__(
            self,
            model_path: str,
            device: Literal["cpu", "cuda"] | str = "cpu",
            batch_size: int = 32,
            wait_timeout: float = 0.01):

        # Initialize device and vocoder
        self.device = torch.device(device)

        self.vocoder = BiCodec.load_from_checkpoint(model_path)
        del self.vocoder.encoder, self.vocoder.postnet
        self.vocoder.eval().to(self.device)  # Set model to evaluation mode
        self._engine = AsyncBatchEngine(
            processing_function=self._batch_process,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )

    async def _batch_process(self, requests: list[dict[str, torch.Tensor]]):
        global_tokens_list, semantic_tokens_list = [], []

        # Process each request in batch
        for request in requests:
            global_tokens_list.append(request["global_tokens"].to(self.device))
            semantic_tokens_list.append(request["semantic_tokens"].to(self.device))

        # Concatenate tokens for batch processing
        global_tokens = torch.cat(global_tokens_list, dim=0)
        semantic_tokens = torch.cat(semantic_tokens_list, dim=0)

        # Generate waveforms
        with torch.no_grad():
            wavs = self.vocoder.detokenize(semantic_tokens, global_tokens.unsqueeze(1))

        # Prepare responses
        responses = []
        for i in range(len(requests)):
            responses.append({
                "waveform": wavs[i],
            })

        return responses

    async def _async_process(
            self,
            request: dict[str, torch.Tensor],
            request_id: str
    ) -> list[float]:
        embedding = await self._engine.add_request(
            single_input=request, request_id=request_id
        )
        return embedding.get("feature")

    async def async_process(
            self,
            request: dict[str, torch.Tensor],
            request_id: Optional[str] = None):
        if request_id is None:
            request_id = str(uuid.uuid4().hex)
        output = await self._async_process(
            request=request,
            request_id=request_id)

        return output
