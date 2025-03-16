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
# 临时存放合成音频的目录
TEMP_DIR = "TTS-TEMP"
# 角色克隆的音频文件列表，如需添加，请添加到列表中
ROLE_WAVS = [
    {
        "name": "哪吒",
        "reference_audio": "example/roles/nezha.wav",
        "reference_text": "别烦我，让我一个人安静地死去。"
    },
    {
        "name": "李靖",
        "reference_audio": "example/roles/lijing.wav",
        "reference_text": "你可知为什么人们都怕你？"
    },
    {
        "name": "殷夫人",
        "reference_audio": "example/roles/yin_furen.wav",
        "reference_text": "出去跟爹娘一起斩妖除魔，为民除害。"
    },
    {
        "name": "后羿",
        "reference_audio": "example/roles/houyi.wav",
        "reference_text": "周日被我射熄火了，所以今天是周一。"
    }
]

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

ROLE_MAPPING = {}
ROLES = []
ROLE_INFO_LIST = []
for i, role in enumerate(ROLE_WAVS):
    uid = str(i)
    if role['name'] in ROLES:
        print(f"{role['name']}角色已存在")
        continue
    try:
        with open(role["reference_audio"], "rb") as f:
            audio_bytes = f.read()
        # 将二进制音频数据转换为 base64 字符串
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"读取{role['name']}音频失败：", e)
        continue
    ROLES.append(role['name'])
    ROLE_INFO_LIST.append({
        "role_id": uid,
        "name": role['name']
    })
    ROLE_MAPPING[uid] = {
        "reference_text": role["reference_text"],
        "reference_audio": audio_base64,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/audio_roles', methods=['GET'])
def audio_roles():
    return jsonify({
        "success": True,
        "roles": ROLE_INFO_LIST
    })


@app.route('/generate_voice', methods=['POST'])
def generate_voice():
    # 从前端获取参数
    data = request.form
    payload = {
        "text": data.get('text', ''),
        "gender": data.get('gender', 'male'),
        "pitch": data.get('pitch', 'moderate'),
        "speed": data.get('speed', 'moderate'),
        "temperature": float(data.get('temperature', 0.6)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 2048))
    }

    # 调用API
    try:
        response = requests.post(f"{BASE_URL}/generate_voice", json=payload)
        response.raise_for_status()

        # 创建临时文件保存音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP_DIR)
        temp_file.write(response.content)
        temp_file.close()
        return jsonify({
            "success": True,
            "message": "语音生成成功",
            "file_path": os.path.basename(temp_file.name)
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
        "temperature": float(data.get('temperature', 0.6)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 2048))
    }

    # 调用API
    try:
        response = requests.post(f"{BASE_URL}/clone_voice", json=payload)
        response.raise_for_status()

        # 创建临时文件保存音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP_DIR)
        temp_file.write(response.content)
        temp_file.close()

        return jsonify({
            "success": True,
            "message": "声音克隆成功",
            "file_path": os.path.basename(temp_file.name)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"请求失败: {str(e)}"
        }), 500


@app.route('/clone_by_role', methods=['POST'])
def clone_by_role():
    # 从前端获取参数
    data = request.form
    role_id = data.get('role_id', None)
    if role_id is None or role_id not in ROLE_MAPPING:
        return jsonify({
            "success": False,
            "message": "未提供角色ID或角色ID无效"
        }), 400

    payload = {
        "text": data.get('text', ''),
        "reference_text": ROLE_MAPPING[role_id]['reference_text'],
        "reference_audio": ROLE_MAPPING[role_id]['reference_audio'],
        "temperature": float(data.get('temperature', 0.6)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 2048))
    }

    # 调用API
    try:
        response = requests.post(f"{BASE_URL}/clone_voice", json=payload)
        response.raise_for_status()

        # 创建临时文件保存音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP_DIR)
        temp_file.write(response.content)
        temp_file.close()

        return jsonify({
            "success": True,
            "message": "声音克隆成功",
            "file_path": os.path.basename(temp_file.name)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"请求失败: {str(e)}"
        }), 500


@app.route('/play_audio/<path:file_path>')
def play_audio(file_path):
    try:
        file_path = os.path.join(TEMP_DIR, file_path)

        if os.path.exists(file_path):
            return send_file(file_path, mimetype='audio/wav')
        else:
            return jsonify({
                "success": False,
                "message": "文件不存在"
            }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"播放音频失败: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001)
