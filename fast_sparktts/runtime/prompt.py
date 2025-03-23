# -*- coding: utf-8 -*-
# Time      :2025/3/23 20:30
# Author    :Hui Huang
from typing import Optional, Tuple, Literal

import torch

TASK_TOKEN_MAP = {
    "vc": "<|task_vc|>",
    "tts": "<|task_tts|>",
    "asr": "<|task_asr|>",
    "s2s": "<|task_s2s|>",
    "t2s": "<|task_t2s|>",
    "understand": "<|task_understand|>",
    "caption": "<|task_cap|>",
    "controllable_tts": "<|task_controllable_tts|>",
    "prompt_tts": "<|task_prompt_tts|>",
    "speech_edit": "<|task_edit|>",
}

LEVELS_MAP = {
    "very_low": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}

GENDER_MAP = {
    "female": 0,
    "male": 1,
}


def process_prompt(
        text: str,
        prompt_text: Optional[str] = None,
        global_token_ids: torch.Tensor = None,
        semantic_token_ids: torch.Tensor = None,
) -> Tuple[str, torch.Tensor]:
    """
    Process input for voice cloning.

    Args:
        text: The text input to be converted to speech.
        prompt_text: Transcript of the prompt audio.
        global_token_ids: Global token IDs extracted from reference audio.
        semantic_token_ids: Semantic token IDs extracted from reference audio.

    Returns:
        Tuple containing the formatted input prompt and global token IDs.
    """
    # Convert global tokens to string format
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )

    # Prepare the input tokens for the model
    if prompt_text is not None and len(prompt_text) > 0:
        # Include semantic tokens when prompt text is provided
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
        )

        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            prompt_text,
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
            "<|start_semantic_token|>",
            semantic_tokens,
        ]
    else:
        # Without prompt text, exclude semantic tokens
        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
        ]

    # Join all input components into a single string
    inputs = "".join(inputs)
    return inputs, global_token_ids


def process_prompt_control(
        text: str,
        gender: Optional[Literal["female", "male"]] = "female",
        pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
        speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
):
    """
    Process input for voice creation.

    Args:
        gender (str): female | male.
        pitch (str): very_low | low | moderate | high | very_high
        speed (str): very_low | low | moderate | high | very_high
        text (str): The text input to be converted to speech.

    Return:
        str: Input prompt
    """
    assert gender in GENDER_MAP.keys()
    assert pitch in LEVELS_MAP.keys()
    assert speed in LEVELS_MAP.keys()

    gender_id = GENDER_MAP[gender]
    pitch_level_id = LEVELS_MAP[pitch]
    speed_level_id = LEVELS_MAP[speed]

    pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
    speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
    gender_tokens = f"<|gender_{gender_id}|>"

    attribte_tokens = "".join(
        [gender_tokens, pitch_label_tokens, speed_label_tokens]
    )

    control_tts_inputs = [
        TASK_TOKEN_MAP["controllable_tts"],
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_style_label|>",
        attribte_tokens,
        "<|end_style_label|>",
    ]

    return "".join(control_tts_inputs)
