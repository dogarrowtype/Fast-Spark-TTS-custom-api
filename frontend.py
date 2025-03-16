# -*- coding: utf-8 -*-
# Time      :2025/3/16 12:19
# Author    :Hui Huang
import os

from flask import Flask, render_template, request, jsonify, send_file
import requests
import base64
import tempfile

app = Flask(__name__)

# 设置API地址
BASE_URL = "http://localhost:8000"
TEMP_DIR = "TTS-TEMP"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_voice', methods=['POST'])
def generate_voice():
    # 从前端获取参数
    data = request.form
    payload = {
        "text": data.get('text', ''),
        "gender": data.get('gender', 'male'),
        "pitch": data.get('pitch', 'moderate'),
        "speed": data.get('speed', 'moderate'),
        "temperature": float(data.get('temperature', 0.2)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 512))
    }

    # 调用API
    try:
        response = requests.post(f"{BASE_URL}/generate_voice", json=payload)
        response.raise_for_status()

        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR, exist_ok=True)
        # 创建临时文件保存音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP_DIR)
        temp_file.write(response.content)
        temp_file.close()

        return jsonify({
            "success": True,
            "message": "语音生成成功",
            "file_path": temp_file.name
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"请求失败: {str(e)}"
        }), 500


@app.route('/clone_voice', methods=['POST'])
def clone_voice():
    # 从前端获取参数
    data = request.form
    audio_file = request.files.get('reference_audio')

    # 读取并编码音频文件
    if audio_file:
        audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    else:
        return jsonify({
            "success": False,
            "message": "未提供参考音频文件"
        }), 400

    payload = {
        "text": data.get('text', ''),
        "reference_text": data.get('reference_text', ''),
        "reference_audio": audio_base64,
        "temperature": float(data.get('temperature', 0.2)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 512))
    }

    # 调用API
    try:
        response = requests.post(f"{BASE_URL}/clone_voice", json=payload)
        response.raise_for_status()

        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR, exist_ok=True)

        # 创建临时文件保存音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP_DIR)
        temp_file.write(response.content)
        temp_file.close()

        return jsonify({
            "success": True,
            "message": "声音克隆成功",
            "file_path": temp_file.name
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"请求失败: {str(e)}"
        }), 500


@app.route('/play_audio/<path:file_path>')
def play_audio(file_path):
    try:
        return send_file(file_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"播放音频失败: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001)
