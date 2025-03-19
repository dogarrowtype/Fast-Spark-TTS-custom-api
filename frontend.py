# -*- coding: utf-8 -*-
# Time      :2025/3/16 12:19
# Author    :Hui Huang
import argparse
import os

from flask import Flask, render_template, request, jsonify, send_file
import requests
import base64
import tempfile
from fast_sparktts.runtime.logger import get_logger, setup_logging

logger = get_logger()

app = Flask(__name__)

ROLE_MAPPING = {}
ROLES = []
ROLE_INFO_LIST = []


def load_roles(role_dir: str):
    if not os.path.exists(role_dir):
        logger.warning("当前角色目录不存在，将无法使用角色克隆功能！")
        return

    role_list = os.listdir(role_dir)
    for i, role in enumerate(role_list):
        uid = str(i)
        if role in ROLES:
            logger.warning(f"{role}角色已存在")
            continue
        try:
            with open(os.path.join(role_dir, role, "reference_audio.wav"), "rb") as f:
                audio_bytes = f.read()
            # 将二进制音频数据转换为 base64 字符串
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            logger.warning(f"读取{role}音频失败：", e)
            continue

        try:
            role_text = open(os.path.join(role_dir, role, "reference_text.txt"), "r", encoding='utf8').read()
            role_text = role_text.strip()
        except Exception as e:
            logger.warning(f"读取{role}文本失败：", e)
            role_text = None

        if role_text == "":
            logger.warning(f"{role}角色文本为空")
            role_text = None

        ROLES.append(role)
        ROLE_INFO_LIST.append({
            "role_id": uid,
            "name": role
        })
        ROLE_MAPPING[uid] = {
            "reference_text": role_text,
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
        response = requests.post(f"{args.backend_url}/generate_voice", json=payload)
        response.raise_for_status()

        # 创建临时文件保存音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=args.audio_dir)
        temp_file.write(response.content)
        temp_file.close()

        return jsonify({
            "success": True,
            "message": "语音生成成功",
            "file_path": os.path.basename(temp_file.name)
        })
    except Exception as e:
        logger.warning(f"请求语音合成接口失败: {str(e)}")
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
        logger.error("用户未上传参考音频文件")
        return jsonify({
            "success": False,
            "message": "未提供参考音频文件"
        }), 400

    payload = {
        "text": data.get('text', ''),
        "reference_text": data.get('reference_text', None),
        "reference_audio": audio_base64,
        "temperature": float(data.get('temperature', 0.6)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 2048))
    }

    # 调用API
    try:
        response = requests.post(f"{args.backend_url}/clone_voice", json=payload)
        response.raise_for_status()

        # 创建临时文件保存音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=args.audio_dir)
        temp_file.write(response.content)
        temp_file.close()
        return jsonify({
            "success": True,
            "message": "声音克隆成功",
            "file_path": os.path.basename(temp_file.name)
        })
    except Exception as e:
        logger.warning(f"请求声音克隆接口失败: {str(e)}")
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
        logger.warning("用户未提供角色ID或角色ID无效")
        return jsonify({
            "success": False,
            "message": "未提供角色ID或角色ID无效"
        }), 400

    payload = {
        "text": data.get('text', ''),
        "reference_text": ROLE_MAPPING[role_id].get('reference_text', None),
        "reference_audio": ROLE_MAPPING[role_id]['reference_audio'],
        "temperature": float(data.get('temperature', 0.6)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 2048))
    }

    # 调用API
    try:
        response = requests.post(f"{args.backend_url}/clone_voice", json=payload)
        response.raise_for_status()

        # 创建临时文件保存音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=args.audio_dir)
        temp_file.write(response.content)
        temp_file.close()
        return jsonify({
            "success": True,
            "message": "声音克隆成功",
            "file_path": os.path.basename(temp_file.name)
        })
    except Exception as e:
        logger.warning(f"角色克隆请求后端接口失败：{e}")
        return jsonify({
            "success": False,
            "message": f"请求失败: {str(e)}"
        }), 500


@app.route('/play_audio/<path:file_path>')
def play_audio(file_path):
    try:
        file_path = os.path.join(args.audio_dir, file_path)

        if os.path.exists(file_path):
            return send_file(file_path, mimetype='audio/wav')
        else:
            logger.warning("请求播放的文件不存在")
            return jsonify({
                "success": False,
                "message": "文件不存在"
            }), 404
    except Exception as e:
        logger.warning(f"播放音频失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"播放音频失败: {str(e)}"
        }), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FastSparkTTS 前端")
    parser.add_argument("--backend_url", type=str, default="http://localhost:8000",
                        help="FastSparkTTS服务端接口地址")
    parser.add_argument("--audio_dir", type=str, default="data/audios",
                        help="保存合成语音临时文件的目录")
    parser.add_argument("--role_dir", type=str, default="data/roles",
                        help="存放已有角色信息的目录")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="前端地址")
    parser.add_argument("--port", type=int, default=8001, help="前端端口")
    args = parser.parse_args()

    setup_logging()

    logger.info("启动FastSparkTTS前端服务")
    logger.info(f"Config: {args}")

    if not os.path.exists(args.audio_dir):
        os.makedirs(args.audio_dir, exist_ok=True)
        logger.info(f"创建临时文件目录：{args.audio_dir}")

    load_roles(args.role_dir)

    app.run(host=args.host, port=args.port)
