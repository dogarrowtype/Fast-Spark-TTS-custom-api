# -*- coding: utf-8 -*-
# Time      :2025/3/16 12:19
# Author    :Hui Huang
import argparse
from flask import Flask, render_template, request, jsonify, Response
import httpx
import base64
from fast_tts import get_logger, setup_logging

logger = get_logger()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/audio_roles', methods=['GET'])
def audio_roles():
    try:
        r = httpx.get(f"{args.backend_url}/audio_roles", timeout=None)
        r.raise_for_status()
        result = r.json()
        return jsonify({
            "success": True,
            "roles": [{"role_id": role, "name": role} for role in result['roles']]
        })
    except Exception as e:
        logger.warning(f"获取角色列表失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"获取角色列表失败: {str(e)}"
        }), 500


@app.route('/generate_voice', methods=['POST'])
def generate_voice():
    data = request.form

    payload = {
        "text": data.get('text', ''),
        "gender": data.get('gender', 'male'),
        "pitch": data.get('pitch', 'moderate'),
        "speed": data.get('speed', 'moderate'),
        "temperature": float(data.get('temperature', 0.9)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 4096))
    }

    try:
        r = httpx.post(f"{args.backend_url}/generate_voice", json=payload, timeout=None)
        r.raise_for_status()
        return Response(r.content, mimetype='audio/wav')
    except Exception as e:
        logger.warning(f"请求语音合成接口失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"请求失败: {str(e)}"
        }), 500


@app.route('/clone_voice', methods=['POST'])
def clone_voice():
    data = request.form
    audio_file = request.files.get('reference_audio')
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
        "temperature": float(data.get('temperature', 0.9)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 4096))
    }

    try:
        r = httpx.post(f"{args.backend_url}/clone_voice", json=payload, timeout=None)
        r.raise_for_status()
        return Response(r.content, mimetype='audio/wav')
    except Exception as e:
        logger.warning(f"请求声音克隆接口失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"请求失败: {str(e)}"
        }), 500


@app.route('/clone_by_role', methods=['POST'])
def clone_by_role():
    data = request.form

    role_id = data.get('role_id', None)
    if role_id is None:
        logger.warning("用户未提供角色ID或角色ID无效")
        return jsonify({
            "success": False,
            "message": "未提供角色ID或角色ID无效"
        }), 400

    payload = {
        "name": role_id,
        "text": data.get('text', ''),
        "temperature": float(data.get('temperature', 0.9)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 4096))
    }

    try:

        r = httpx.post(f"{args.backend_url}/speak", json=payload, timeout=None)
        r.raise_for_status()
        return Response(r.content, mimetype='audio/wav')
    except Exception as e:
        logger.warning(f"角色音频合成失败：{e}")
        return jsonify({
            "success": False,
            "message": f"角色音频合成失败: {str(e)}"
        }), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FastSparkTTS 前端")
    parser.add_argument("--backend_url", type=str, default="http://127.0.0.1:8000",
                        help="FastSparkTTS服务端接口地址")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="前端地址")
    parser.add_argument("--port", type=int, default=8001, help="前端端口")
    args = parser.parse_args()

    setup_logging()

    logger.info("启动FastSparkTTS前端服务")
    logger.info(f"Config: {args}")

    app.run(host=args.host, port=args.port)
