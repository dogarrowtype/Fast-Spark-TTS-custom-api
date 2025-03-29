# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/27 9:19
# Author  : Hui Huang
import argparse
import base64
import gradio as gr
import httpx
import numpy as np

stream_mode_list = [('否', False), ('是', True)]


def encode_bs64(audio_path):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    # 将二进制音频数据转换为 base64 字符串
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_base64


def get_speakers() -> list[str]:
    r = httpx.get(f"{args.backend_url}/audio_roles", timeout=None)
    r.raise_for_status()
    result = r.json()
    return result


def get_headers():
    if args.api_key is not None:
        headers = {
            "Authorization": f"Bearer {args.api_key}",
        }
    else:
        headers = None
    return headers


def stream_generate(url, payload):
    with httpx.stream("POST", url, json=payload, timeout=None, headers=get_headers()) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            if chunk:
                audio_chunk = np.frombuffer(chunk, dtype=np.int16)
                yield 16000, audio_chunk


def generate(url, payload):
    r = httpx.post(url, json=payload, timeout=None, headers=get_headers())
    r.raise_for_status()
    return 16000, np.frombuffer(r.content, dtype=np.int16)


def generate_voice(text, gender, pitch, speed, stream):
    num2attr = ["very_low", "low", "moderate", "high", "very_high"]
    payload = {
        "text": text,
        "gender": gender,
        "pitch": num2attr[pitch],
        "speed": num2attr[speed],
        "stream": stream,
    }
    # 如果想取消流式播放，修改如下
    # audio = generate(f"{args.backend_url}/generate_voice", payload)
    # return audio
    if stream:
        for chunk in stream_generate(f"{args.backend_url}/generate_voice", payload):
            yield chunk
    else:
        audio = generate(f"{args.backend_url}/generate_voice", payload)
        yield audio


def clone_voice(text, audio_upload, audio_record, stream):
    if audio_upload is not None:
        reference_audio = audio_upload
    elif audio_record is not None:
        reference_audio = audio_record
    else:
        raise TypeError("audio_record or audio_upload must be provided")

    payload = {
        "text": text,
        "reference_text": None,
        "reference_audio": encode_bs64(reference_audio),
        "stream": stream,
    }
    # 如果想取消流式播放，修改如下
    # audio = generate(f"{args.backend_url}/clone_voice", payload)
    # return audio
    if stream:
        return stream_generate(f"{args.backend_url}/clone_voice", payload)
    else:
        audio = generate(f"{args.backend_url}/clone_voice", payload)
        yield audio


def build_ui():
    with gr.Blocks() as demo:
        # Use HTML for centered title
        gr.HTML('<h1 style="text-align: center;">Fast Spark TTS</h1>')
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
                stream = gr.Radio(choices=stream_mode_list, label='Whether to use streaming inference',
                                  value=stream_mode_list[0][1])

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )

                generate_buttom_clone = gr.Button("Generate")

                generate_buttom_clone.click(
                    clone_voice,
                    inputs=[
                        text_input,
                        prompt_wav_upload,
                        prompt_wav_record,
                        stream
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

                stream = gr.Radio(choices=stream_mode_list, label='Whether to use streaming inference',
                                  value=stream_mode_list[0][1])
                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )
                create_button.click(
                    generate_voice,
                    inputs=[text_input_creation, gender, pitch, speed, stream],
                    outputs=[audio_output],
                )

    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradio Web UI")
    parser.add_argument("--backend_url", type=str, default="http://127.0.0.1:8000",
                        help="FastSparkTTS服务端接口地址")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="前端地址")
    parser.add_argument("--port", type=int, default=7860, help="前端端口")
    parser.add_argument("--api_key", type=str, default=None, help="后端接口访问的api key")
    args = parser.parse_args()

    demo = build_ui()

    # Launch Gradio with the specified server name and port
    demo.launch(
        server_name=args.host,
        server_port=args.port
    )
