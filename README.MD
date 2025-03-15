Fast-Spark-TTS
---

## Overview

使用vllm和llama cpp加速spark tts本地合成速度，支持CPU处理。

## Install

1. pip install torch==2.5.1 torchaudio==2.5.1

2. pip install -r requirements.txt

## 下载地址

权重下载地址：[huggingface](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)、[modelscope](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)

## 推理速度

显卡：A800

使用[prompt_audio.wav](example/prompt_audio.wav)测试克隆速度，总共循环推理五次，计算平均推理时间 (s)。

测试代码可参考[speed_test.py](speed_test.py)

使用vllm后，主要耗时不在llm，而在audio tokenizer和vocoder，应该可以使用onnx进一步优化。

|  engine   | device | Avg Time | 	Avg Time (warm up) |
|:---------:|:------:|:--------:|:-------------------:|
| Official  |  CPU   |  27.20   |        27.30        |
| Official  |  GPU   |   5.95   |        4.97         |
| llama-cpp |  CPU   |  11.32   |        11.09        |
|   vllm    |  GPU   |   1.95   |        1.22         |
|  sglang   |  GPU   |   3.41   |        0.76         |

## Basic Usage

### 1. 本地推理

使用方法参考 [inference.py](inference.py)

接口部署，循环推理建议使用async异步方法。

请注意：vllm，sglang等后端第一次推理会比较耗时，后面速度就会正常。如果评测，建议先使用第一条数据warmup。

### 2. Gradio WebUI

```bash
pip install gradio
```

自行修改[webui.py](webui.py)配置参数，修改完后运行：

```bash
python webui.py
```

### 部署接口

```bash
pip install fastapi uvicorn httpx
```

```bash
python server.py \
--model_path Spark-TTS-0.5B \
--engine vllm \
--llm_device cuda \
--audio_device cuda \
--vocoder_device cuda \
--wav2vec_attn_implementation sdpa \
--max_length 32768 \
--llm_gpu_memory_utilization 0.6 \
--host 0.0.0.0 \
--port 8000
```
接口请求参考[client.py](client.py)


### 依赖说明
#### vLLM 安装方式

vllm版本需要大于`0.7.2`

```bash
pip install vllm
```

具体参考链接：https://github.com/vllm-project/vllm

#### llama-cpp 配置方式

```bash
pip install llama-cpp-python
```

将LLM权重转为gguf格式，然后文件名保存为`model.gguf`，放至`LLM`路径。可以参考下面方式转换权重，如需量化可以自行配置参数。

```bash
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp

python convert_hf_to_gguf.py Spark-TTS-0.5B/LLM --outfile Spark-TTS-0.5B/LLM/model.gguf
```

#### sglang 配置方式

```bash
pip install sglang
```

具体参考链接：https://github.com/sgl-project/sglang

### Reference

1. [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)