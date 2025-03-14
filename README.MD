Fast-Spark-TTS
---

### Overview

使用vllm和llama cpp加速spark tts本地合成速度，支持CPU处理。

### Install

1. pip install torch==2.5.1 torchaudio==2.5.1

2. pip install -r requirements.txt

### 推理速度

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

### Basic Usage

demo可以参考 [inference.py](inference.py)

异步方法demo参考 [infer_async.py](infer_async.py)

请注意：vllm，sglang等后端第一次推理会比较耗时，后面速度就会正常。如果评测，建议先使用第一条数据warmup。

权重下载地址：[huggingface](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)、[modelscope](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)

#### vLLM 配置方式

```bash
pip install vllm
```

修改[inference.py](inference.py)部分代码为：

```python
predictor = FastSparkTTS(
    model_path="Spark-TTS-0.5B",  # 模型基础路径
    max_length=32768,  # 最大上下文长度
    llm_device="cuda:0",  # vllm device
    bicodec_device="cuda:0",  # 语音模块device
    backend="vllm",
    attn_implementation="sdpa",  # 使用flash attn加速wav2vec
    gpu_memory_utilization=0.5  # vllm 显存占用比例，分一半给语音模型
)
```

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

修改[inference.py](inference.py)部分代码为：

```python
predictor = FastSparkTTS(
    model_path="Spark-TTS-0.5B",  # 模型基础路径，LLM下需要有model.gguf文件
    max_length=32768,  # 最大上下文长度
    llm_device="cpu",  # vllm device
    bicodec_device="cpu",  # 语音模块device
    backend="llama-cpp",
    attn_implementation=None,  # 使用flash attn加速wav2vec
)
```

#### gradio web 启动

```bash
pip install gradio
```

按照Basic Usage修改[webui.py](webui.py)

```bash
python webui.py
```

### Reference

1. [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)