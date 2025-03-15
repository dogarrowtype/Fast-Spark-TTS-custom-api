# -*- coding: utf-8 -*-
# Time      :2025/3/15 13:39
# Author    :Hui Huang

import requests
import base64

# 设置服务器地址
BASE_URL = "http://localhost:8000"

def generate_voice():
    text = "你是谁。"
    payload = {
        "text": text,
        "gender": "male",
        "pitch": "moderate",
        "speed": "moderate",
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 512
    }
    response = requests.post(f"{BASE_URL}/generate_voice", json=payload)
    if response.status_code == 200:
        with open("generate_voice.wav", "wb") as f:
            f.write(response.content)
        print("生成的音频已保存为 generate_voice.wav")
    else:
        print("请求失败：", response.status_code, response.text)

def clone_with_base64():
    # 使用 base64 编码的参考音频
    text = "你是谁。"
    reference_text = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"

    reference_audio_path = "example/prompt_audio.wav"  # 请替换为你本地的参考音频文件路径
    try:
        with open(reference_audio_path, "rb") as f:
            audio_bytes = f.read()
        # 将二进制音频数据转换为 base64 字符串
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print("读取本地文件失败：", e)
        return

    payload = {
        "text": text,
        "reference_text": reference_text,
        "reference_audio": audio_base64,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 512
    }
    response = requests.post(f"{BASE_URL}/clone_voice", json=payload)
    if response.status_code == 200:
        with open("clone_voice.wav", "wb") as f:
            f.write(response.content)
        print("克隆的音频已保存为 clone_voice.wav")
    else:
        print("请求失败：", response.status_code, response.text)


if __name__ == "__main__":
    generate_voice()
