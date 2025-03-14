# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/13 13:18
# Author  : Hui Huang
import os
from typing import Literal, Optional
import soundfile as sf
import gradio as gr

from datetime import datetime
from fast_sparktts import AsyncFastSparkTTS
from fast_sparktts.utils.token_parser import LEVELS_MAP_UI


def save_audio(
        audio, save_dir="example/results"):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")
    sf.write(save_path, audio, samplerate=16000)
    return save_path


def build_ui(
        model_path: str,
        max_length: int = 32768,
        gguf_model_file: Optional[str] = None,
        llm_device: Literal["cpu", "cuda"] | str = "cpu",
        audio_device: Literal["cpu", "cuda"] | str = "cpu",
        vocoder_device: Literal["cpu", "cuda"] | str = "cpu",
        engine: Literal["vllm", "llama-cpp", "sglang"] = "llama-cpp",
        wav2vec_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
        llm_gpu_memory_utilization: Optional[float] = 0.6,
        batch_size: int = 32,
        wait_timeout: float = 0.01,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        max_tokens=1024,
        **kwargs, ):
    # Initialize model
    model = AsyncFastSparkTTS(
        model_path=model_path,
        max_length=max_length,
        gguf_model_file=gguf_model_file,
        llm_device=llm_device,
        audio_device=audio_device,
        vocoder_device=vocoder_device,
        engine=engine,
        wav2vec_attn_implementation=wav2vec_attn_implementation,
        llm_gpu_memory_utilization=llm_gpu_memory_utilization,
        batch_size=batch_size,
        wait_timeout=wait_timeout,
        **kwargs,
    )

    # Define callback function for voice cloning
    async def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record):
        """
        Gradio callback to clone voice using text and optional prompt speech.
        - text: The input text to be synthesised.
        - prompt_text: Additional textual info for the prompt (optional).
        - prompt_wav_upload/prompt_wav_record: Audio files used as reference.
        """
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text_clean = None if len(prompt_text) < 2 else prompt_text

        audio = await model.async_clone_voice(
            reference_wav_path=prompt_speech,
            reference_text=prompt_text_clean,
            target_text=text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens
        )
        audio_output_path = save_audio(audio)
        return audio_output_path

    # Define callback function for creating new voices
    async def voice_creation(text, gender, pitch, speed):
        """
        Gradio callback to create a synthetic voice with adjustable parameters.
        - text: The input text for synthesis.
        - gender: 'male' or 'female'.
        - pitch/speed: Ranges mapped by LEVELS_MAP_UI.
        """
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]
        audio = await model.async_generate_voice(
            text=text,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens
        )
        audio_output_path = save_audio(audio)
        return audio_output_path

    with gr.Blocks() as demo:
        # Use HTML for centered title
        gr.HTML('<h1 style="text-align: center;">Spark-TTS by SparkAudio</h1>')
        with gr.Tabs():
            # Voice Clone Tab
            with gr.TabItem("Voice Clone"):
                gr.Markdown(
                    "### Upload reference audio or recording （上传参考音频或者录音）"
                )

                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="Choose the prompt audio file, ensuring the sampling rate is no lower than 16kHz.",
                    )
                    prompt_wav_record = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Record the prompt audio file.",
                    )

                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text", lines=3, placeholder="Enter text here"
                    )
                    prompt_text_input = gr.Textbox(
                        label="Text of prompt speech (Optional; recommended for cloning in the same language.)",
                        lines=3,
                        placeholder="Enter text of the prompt speech.",
                    )

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )

                generate_buttom_clone = gr.Button("Generate")

                generate_buttom_clone.click(
                    voice_clone,
                    inputs=[
                        text_input,
                        prompt_text_input,
                        prompt_wav_upload,
                        prompt_wav_record,
                    ],
                    outputs=[audio_output],
                )

            # Voice Creation Tab
            with gr.TabItem("Voice Creation"):
                gr.Markdown(
                    "### Create your own voice based on the following parameters"
                )

                with gr.Row():
                    with gr.Column():
                        gender = gr.Radio(
                            choices=["male", "female"], value="male", label="Gender"
                        )
                        pitch = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Pitch"
                        )
                        speed = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Speed"
                        )
                    with gr.Column():
                        text_input_creation = gr.Textbox(
                            label="Input Text",
                            lines=3,
                            placeholder="Enter text here",
                            value="You can generate a customized voice by adjusting parameters such as pitch and speed.",
                        )
                        create_button = gr.Button("Create Voice")

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )
                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed],
                    outputs=[audio_output],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui(
        model_path="D:\project\Spark-TTS-Fast\checkpoints",
        max_length=32768,
        llm_device="cpu",
        audio_device="cpu",
        vocoder_device="cpu",
        engine="llama-cpp",
        wav2vec_attn_implementation="eager",
        max_tokens=512,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    # Launch Gradio with the specified server name and port
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )
