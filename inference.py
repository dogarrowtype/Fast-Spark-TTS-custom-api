# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/13 12:32
# Author  : Hui Huang
from sparktts.inference.predictor import FastSparkTTS
import soundfile as sf


def main():
    predictor = FastSparkTTS(
        model_path="D:\project\Spark-TTS-Fast\checkpoints",
        max_length=32768,
        llm_device="cpu",
        bicodec_device="cpu",
        backend="llama-cpp",
    )
    wav = predictor.inference(
        "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
        prompt_speech_path="example/prompt_audio.wav",
        prompt_text="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_tokens=1024
    )
    sf.write("result.wav", wav, samplerate=16000)


if __name__ == '__main__':
    main()
