# -*- coding: utf-8 -*-
# Time      :2025/3/16 12:19
# Author    :Hui Huang
import argparse
import os
import struct

from flask import Flask, render_template, request, jsonify, Response
import httpx
import base64
from fast_tts import get_logger, setup_logging
import uuid

logger = get_logger()

app = Flask(__name__)

AUDIO_MAP = {}


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


def get_headers():
    if args.api_key is not None:
        headers = {
            "Authorization": f"Bearer {args.api_key}",
        }
    else:
        headers = None
    return headers


def create_wav_header(sample_rate=16000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header


def stream_generate(url, payload):
    is_first = True
    with httpx.stream("POST", url, json=payload, timeout=None, headers=get_headers()) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            if chunk:
                if is_first:
                    yield create_wav_header()
                    is_first = False
                yield chunk


def generate(url, payload):
    try:
        r = httpx.post(url, json=payload, timeout=None, headers=get_headers())
        r.raise_for_status()
        return Response(r.content, mimetype='audio/wav')
    except Exception as e:
        logger.warning(f"请求接口失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"请求失败: {str(e)}"
        }), 500


@app.route('/upload_reference_audio', methods=['POST'])
def upload_reference_audio():
    audio_file = request.files.get('reference_audio')

    # 检查文件名是否为空
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': '空文件名'}), 400
    allowed_extensions = {'.wav'}
    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({'success': False, 'error': '不支持的文件格式'}), 400

    if audio_file:
        audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    else:
        logger.error("用户未上传参考音频文件")
        return jsonify({
            "success": False,
            "message": "未提供参考音频文件"
        }), 400
    audio_id = str(uuid.uuid4().hex)
    AUDIO_MAP[audio_id] = audio_base64
    return jsonify({
        "success": True,
        "audio_id": audio_id
    })


@app.route('/generate_voice', methods=['GET'])
def generate_voice():
    data = request.args
    stream = data.get("stream", False)
    if stream == 'true' or stream is True:
        stream = True
    else:
        stream = False

    payload = {
        "text": data.get('text', ''),
        "gender": data.get('gender', 'male'),
        "pitch": data.get('pitch', 'moderate'),
        "speed": data.get('speed', 'moderate'),
        "temperature": float(data.get('temperature', 0.9)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 4096)),
        "stream": stream
    }
    if stream:
        return Response(stream_generate(f"{args.backend_url}/generate_voice", payload), mimetype='audio/wav')

    else:
        return generate(f"{args.backend_url}/generate_voice", payload)


@app.route('/clone_voice', methods=['GET'])
def clone_voice():
    data = request.args
    stream = data.get("stream", False)
    if stream == 'true' or stream is True:
        stream = True
    else:
        stream = False

    audio_id = data.get("audio_id", None)
    if audio_id is None or audio_id not in AUDIO_MAP:
        logger.warning("用户未提供音频ID或音频ID无效")
        return jsonify({
            "success": False,
            "message": "未提供音频ID或音频ID无效"
        }), 400

    payload = {
        "text": data.get('text', ''),
        "reference_text": data.get('reference_text', None),
        "reference_audio": AUDIO_MAP.pop(audio_id),
        "temperature": float(data.get('temperature', 0.9)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 4096)),
        "stream": stream
    }
    if stream:
        return Response(stream_generate(f"{args.backend_url}/clone_voice", payload), mimetype='audio/wav')

    else:
        return generate(f"{args.backend_url}/clone_voice", payload)


@app.route('/clone_by_role', methods=['GET'])
def clone_by_role():
    data = request.args
    stream = data.get("stream", False)
    if stream == 'true' or stream is True:
        stream = True
    else:
        stream = False

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
        "max_tokens": int(data.get('max_tokens', 4096)),
        "stream": stream
    }
    if stream:
        return Response(stream_generate(f"{args.backend_url}/speak", payload), mimetype='audio/wav')

    else:
        return generate(f"{args.backend_url}/speak", payload)


@app.route('/multi_role_speak', methods=['GET'])
def multi_role_speak():
    data = request.args
    stream = data.get("stream", False)
    if stream == 'true' or stream is True:
        stream = True
    else:
        stream = False

    payload = {
        "text": data.get('text', ''),
        "temperature": float(data.get('temperature', 0.9)),
        "top_p": float(data.get('top_p', 0.95)),
        "top_k": int(data.get('top_k', 50)),
        "max_tokens": int(data.get('max_tokens', 4096)),
        "stream": stream
    }
    if stream:
        return Response(stream_generate(f"{args.backend_url}/multi_speak", payload), mimetype='audio/wav')

    else:
        return generate(f"{args.backend_url}/multi_speak", payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FastTTS 前端")
    parser.add_argument("--backend_url", type=str, default="http://127.0.0.1:8000",
                        help="FastTTS服务端接口地址")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="前端地址")
    parser.add_argument("--port", type=int, default=8001, help="前端端口")
    parser.add_argument("--api_key", type=str, default=None, help="后端接口访问的api key")
    args = parser.parse_args()

    setup_logging()

    logger.info("启动FastTTS前端服务")
    logger.info(f"Config: {args}")

    app.run(host=args.host, port=args.port)
