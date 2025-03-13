Fast-Spark-TTS
---

### Overview

使用vllm和llama cpp加速spark tts本地合成速度，支持CPU处理。


### Install
1. pip install torch==2.5.1 torchaudio==2.5.1 

2. pip install -r requirements.txt

### Basic Usage

demo可以参考 [inference.py](inference.py)

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