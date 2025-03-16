# FastSparkTTS 语音合成与克隆平台 🔊

[中文](README.MD) | [English](README_EN.MD)

> 🚀 **FastSparkTTS** - 基于SparkTTS模型，提供高质量中文语音合成与声音克隆服务。通过简单易用的Web界面，让您轻松创建自然逼真的人声，满足各种场景需求。

## ✨ 特性

- 🚀 **多种后端加速**: 支持`vllm`、`sglang`、`llama cpp`多种加速策略
- 🎯 **高并发**: 采用动态批处理，极大提高并发
- 🎛️ **全参数控制**: 音调、语速、音色温度等全方位可调
- 📱 **轻量部署**: 最小依赖，基于Flask和fastapi快速启动
- 🎨 **简洁界面**: 标准的现代化UI

## 🖼️ 界面预览

<div align="center">
  <table>
    <tr>
      <td><img src="source/generate_ui.png" alt="语音合成界面"></td>
      <td><img src="source/clone.png" alt="声音克隆界面"></td>
    </tr>
    <tr>
      <td align="center"><strong>语音合成</strong></td>
      <td align="center"><strong>声音克隆</strong></td>
    </tr>
  </table>
</div>

## 🛠️ 快速开始

### 环境要求

- Python 3.10+
- Flask 2.0+
- fastapi
- vllm 或 sglang 或 llama-cpp

### 安装依赖

```bash
pip install -r requirements.txt
```

推理引擎安装 (按需求安装一个即可)

- **vLLM**

  vllm版本需要大于`0.7.2`
    ```bash
    pip install vllm
    ```
  具体参考链接：https://github.com/vllm-project/vllm


- **llama-cpp**
    ```bash
    pip install llama-cpp-python
    ```
  将LLM权重转为gguf格式，然后文件名保存为`model.gguf`，放至`LLM`路径。可以参考下面方式转换权重，如需量化可以自行配置参数。
    ```bash
    git clone https://github.com/ggml-org/llama.cpp.git
    
    cd llama.cpp
    
    python convert_hf_to_gguf.py Spark-TTS-0.5B/LLM --outfile Spark-TTS-0.5B/LLM/model.gguf
    ```

- **sglang**

    ```bash
    pip install sglang
    ```

  具体参考链接：https://github.com/sgl-project/sglang

### 下载权重

权重下载地址：[huggingface](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)、[modelscope](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)

### 启动

1. 克隆项目仓库

```bash
git clone https://github.com/HuiResearch/Fast-Spark-TTS.git
cd Fast-Spark-TTS
```

2. 启动SparkTTS API服务

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

3. 启动Web界面

```bash
python frontend.py
```

4. 在浏览器中访问

```
http://localhost:8001
```

## 🚀 使用指南

### 语音合成

1. 切换到「语音合成」标签页
2. 输入需要转换为语音的文本
3. 调整性别、音调、语速等参数
4. 点击「生成语音」按钮
5. 等待生成完成后即可播放或下载

### 声音克隆

1. 切换到「声音克隆」标签页
2. 输入目标文本
3. 上传参考音频
4. 输入参考音频对应的文本
5. 调整参数
6. 点击「克隆声音」按钮
7. 等待克隆完成后即可播放或下载

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

## 本地使用

使用方法参考 [inference.py](inference.py)

接口部署，循环推理建议使用async异步方法。

请注意：vllm，sglang等后端第一次推理会比较耗时，后面速度就会正常。如果评测，建议先使用第一条数据warmup。

## Reference

1. [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)

## 许可协议与致谢

本项目基于 [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) 构建，并采用与 SparkTTS
相同的开源许可协议进行分发。详情请参阅原始 [SparkTTS 许可协议](https://github.com/SparkAudio/Spark-TTS/blob/main/LICENSE)。