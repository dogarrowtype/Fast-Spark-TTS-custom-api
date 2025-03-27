# -*- coding: utf-8 -*-
# Time      :2025/3/23 19:06
# Author    :Hui Huang
import os.path
from typing import Literal, Optional

from omegaconf import DictConfig, OmegaConf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from .batch_processor import AsyncBatchEngine
import torch
import torch.nn as nn
import torchaudio.transforms as TT
import numpy as np
import soundfile as sf
from safetensors.torch import load_file
import soxr
from ..logger import get_logger
from ...modules.encoder_decoder.feat_encoder import Encoder
from ...modules.speaker.speaker_encoder import SpeakerEncoder
from ...modules.vq.factorized_vector_quantize import FactorizedVectorQuantize

logger = get_logger()


def load_config(config_path: str) -> DictConfig:
    """Loads a configuration file and optionally merges it with a base configuration.

    Args:
    config_path (Path): Path to the configuration file.
    """
    # Load the initial configuration from the given path
    config = OmegaConf.load(config_path)

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(config["base_config"])
        config = OmegaConf.merge(base_config, config)

    return config


class BaseModel(nn.Module):
    @classmethod
    def from_pretrained(cls, model_path: str):
        config = load_config(os.path.join(model_path, "config.yaml"))['audio_tokenizer']
        model = cls(config)
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.remove_weight_norm()
        return model

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)


class TokenizerModel(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(**config["encoder"])
        self.quantizer = FactorizedVectorQuantize(**config["quantizer"])
        self.speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

        self.mel_transformer = TT.MelSpectrogram(
            config["mel_params"]["sample_rate"],
            config["mel_params"]["n_fft"],
            config["mel_params"]["win_length"],
            config["mel_params"]["hop_length"],
            config["mel_params"]["mel_fmin"],
            config["mel_params"]["mel_fmax"],
            n_mels=config["mel_params"]["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(
            self,
            audio_features: torch.Tensor,
            audio_clip: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        mel = self.mel_transformer(audio_clip).squeeze(1)
        z = self.encoder(audio_features.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))
        return {
            "semantic_tokens": semantic_tokens,
            "global_tokens": global_tokens
        }


class Tokenizer:
    def __init__(
            self,
            model_path: str,
            device: Literal["cpu", "cuda", "mps"] | str = "cpu",
            attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            batch_size: int = 32,
            wait_timeout: float = 0.01,
    ):
        self.device = torch.device(device)
        wav2vec_path = os.path.join(model_path, "wav2vec2-large-xlsr-53")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            wav2vec_path,
            attn_implementation=attn_implementation
        )
        self.wav2vec2.config.output_hidden_states = True
        self.wav2vec2.to(self.device)
        self.wav2vec2.eval()

        self.model = TokenizerModel.from_pretrained(
            os.path.join(model_path, "BiCodec")
        ).to(self.device)

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_path
        )

        self._batch_processor = AsyncBatchEngine(
            processing_function=self.batch_tokenize_async,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )

    @classmethod
    def get_ref_clip(cls, wav: np.ndarray) -> np.ndarray:
        """Extract reference audio clip for speaker embedding.

        Args:
            wav: Input waveform array

        Returns:
            Reference clip of fixed duration
        """
        SAMPLE_RATE = 16000
        REF_SEGMENT_DURATION = 6  # seconds
        LATENT_HOP_LENGTH = 320

        ref_segment_length = (
                int(SAMPLE_RATE * REF_SEGMENT_DURATION)
                // LATENT_HOP_LENGTH
                * LATENT_HOP_LENGTH
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate if input is too short
            repeat_times = ref_segment_length // wav_length + 1
            wav = np.tile(wav, repeat_times)

        return wav[:ref_segment_length]

    def load_audio(self, audio):
        waveform, sr = sf.read(audio)
        if len(waveform.shape) > 1:
            waveform = waveform[:, 0]

        if sr != 16000:
            waveform = soxr.resample(waveform, sr, 16000, quality="VHQ")
            logger.warning("输入参考音频采样率不为16000，已对其自动进行重采样。")

        wav_len = len(waveform)
        waveform = np.array(waveform, dtype=np.float32)
        waveform = waveform.reshape(1, -1).astype(np.float32)

        # Prepare inputs
        wav = waveform[:, :wav_len].squeeze(0)
        wav_ref_clip = self.get_ref_clip(wav)
        return {
            "audio": wav,
            "audio_clip": wav_ref_clip,
        }

    @torch.no_grad()
    def extract_features(self, wav_list: list[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(
            wav_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        output = self.wav2vec2(
            inputs['input_values'],
            attention_mask=inputs['attention_mask']
        )
        features = (
                           output.hidden_states[11] + output.hidden_states[14] + output.hidden_states[16]
                   ) / 3
        return features

    @torch.no_grad()
    def tokenize(self, audios):
        if not isinstance(audios, list):
            audios = [audios]
        wav_list = []
        audio_clip = []
        for audio in audios:
            audio_info = self.load_audio(audio)
            wav_list.append(audio_info["audio"])
            audio_clip.append(torch.from_numpy(audio_info["audio_clip"]))
        audio_clip = torch.stack(audio_clip, dim=0)

        audio_features = self.extract_features(wav_list)

        outputs = self.model(audio_features, audio_clip.to(self.device))

        return outputs

    async def batch_tokenize_async(self, audios: list) -> list[dict[str, torch.Tensor]]:
        tokenized = self.tokenize(audios)
        # Prepare responses
        responses = []
        for i in range(len(audios)):
            responses.append({
                "global_tokens": tokenized["global_tokens"][i],
                "semantic_tokens": tokenized["semantic_tokens"][i]
            })

        return responses

    async def tokenize_async(self, audio) -> dict[str, torch.Tensor]:
        output = await self._batch_processor.add_request(
            single_input=audio
        )
        return output.get("feature")
