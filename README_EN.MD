# FastSparkTTS Speech Synthesis and Cloning Platform 🔊

[中文](README.MD) | [English](README_EN.MD)

> 🚀 **FastSparkTTS** – Based on the SparkTTS model, this platform provides high-quality Chinese speech synthesis and
> voice cloning services. With an easy-to-use web interface, you can effortlessly create natural and realistic human
> voices to suit various scenarios.

## ✨ Features

- 🚀 **Multiple Backend Acceleration Options**: Supports acceleration strategies such as `vllm`, `sglang`, and
  `llama cpp`
- 🎯 **High Concurrency**: Utilizes dynamic batching to significantly boost concurrent processing
- 🎛️ **Full Parameter Control**: Offers comprehensive adjustments for pitch, speech rate, voice timbre temperature, and
  more
- 📱 **Lightweight Deployment**: Minimal dependencies, with rapid startup based on Flask and fastapi
- 🎨 **Clean Interface**: Features a modern, standardized UI
- 🔊 **Long Text Speech Synthesis**: Capable of synthesizing extended texts while maintaining consistent voice timbre
- 🔄 **Streaming Text-to-Speech Support**: Enables real-time synthesis with simultaneous generation and playback, reducing waiting times and enhancing the interactive experience

## 🖼️ Example

https://github.com/user-attachments/assets/42e36399-c1b7-40e1-908b-24cfa603a55f


## 🛠️ Quick Start

### Requirements

- Python 3.10+
- Flask 2.0+
- fastapi
- vllm or sglang or llama-cpp

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Inference Engine Installation

(Install one as needed; if using torch for inference, you can skip this step)

- **vLLM**

  The vllm version should be greater than `0.7.2`
  ```bash
  pip install vllm
  ```
  For more details, please refer to: https://github.com/vllm-project/vllm

- **llama-cpp**
  ```bash
  pip install llama-cpp-python
  ```
  Convert the LLM weights to gguf format, save the file as `model.gguf`, and place it in the `LLM` directory. You can
  refer to the following method for weight conversion. If quantization is needed, you can configure the parameters
  accordingly.
  ```bash
  git clone https://github.com/ggml-org/llama.cpp.git
  
  cd llama.cpp
  
  python convert_hf_to_gguf.py Spark-TTS-0.5B/LLM --outfile Spark-TTS-0.5B/LLM/model.gguf
  ```

- **sglang**
  ```bash
  pip install sglang
  ```
  For more details, please refer to: https://github.com/sgl-project/sglang

### Downloading Weights

Weight download
links: [huggingface](https://huggingface.co/SparkAudio/Spark-TTS-0.5B), [modelscope](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)

### Start

1. **Clone the project repository**

   ```bash
   git clone https://github.com/HuiResearch/Fast-Spark-TTS.git
   cd Fast-Spark-TTS
   ```

2. **Start the SparkTTS API Service**

   The engine can be chosen according to your environment; currently supported options include `torch`, `vllm`,
   `sglang`, and `llama-cpp`.
   ```bash
   python server.py \
   --model_path Spark-TTS-0.5B \
   --backend vllm \
   --llm_device cuda \
   --tokenizer_device cuda \
   --detokenizer_device cuda \
   --wav2vec_attn_implementation sdpa \
   --max_length 32768 \
   --llm_gpu_memory_utilization 0.6 \
   --host 0.0.0.0 \
   --port 8000
   ```

3. **Start the Web Interface**

   ```bash
   python frontend.py \
    --backend_url http://127.0.0.1:8000 \ 
    --host 0.0.0.0 \ 
    --port 7860
   ```

4. **Access via your browser**

   ```
   http://localhost:8001
   ```

## 🚀 User Guide

### Speech Synthesis

1. Switch to the **Speech Synthesis** tab.
2. Enter the text you wish to convert to speech.
3. Adjust parameters such as gender, pitch, and speech rate.
4. Click the **Generate Speech** button.
5. Once generation is complete, play or download the audio.

### Voice Cloning

1. Switch to the **Voice Cloning** tab.
2. Enter the target text.
3. Upload the reference audio.
4. Enter the corresponding text for the reference audio.
5. Adjust the parameters.
6. Click the **Clone Voice** button.
7. Once cloning is complete, play or download the audio.

### Character Cloning

1. Switch to the **Character Cloning** tab.
2. Enter the target text.
3. Choose your desired character.
4. Adjust the parameters.
5. Click the **Character Cloning** button.
6. Once cloning is complete, play or download the audio.

## Inference Speed

- GPU: `A800`
- For testing parameters and evaluation methods, please refer to [speed_test.py](speed_test.py)
- Both output audio length (audio len) and inference time (cost time) are measured in seconds.
- The evaluation includes both long-text and short-text scenarios.
- The official code has not been evaluated; if needed, please conduct your own tests.

|  Scenario  |  engine   | device | audio len | cost time |  RTF  |
|:----------:|:---------:|:------:|:---------:|:---------:|:-----:|
| Short Text | llama-cpp |  cpu   |   7.48    |   6.808   | 0.910 |
| Short Text |   torch   |  gpu   |   7.18    |   7.675   | 1.069 |
| Short Text |   vllm    |  gpu   |   7.24    |   1.664   | 0.230 |
| Short Text |  sglang   |  gpu   |   7.58    |   1.073   | 0.142 |
| Long Text  | llama-cpp |  cpu   |  121.98   |  117.828  | 0.966 |
| Long Text  |   torch   |  gpu   |   113.7   |  107.167  | 0.943 |
| Long Text  |   vllm    |  gpu   |  111.82   |   7.282   | 0.065 |
| Long Text  |  sglang   |  gpu   |  117.02   |   4.197   | 0.036 |

## Local Usage

Usage instructions can be found in [inference.py].

For API deployment and repeated inference calls, it is recommended to use asynchronous (async) methods.

**Note:** For backends like vllm and sglang, the first inference call might take longer, but subsequent calls will
perform normally. For benchmarking, it is advised to warm up using the first data entry.

## Reference

1. [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)

## ⚠️ Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and
lawful applications such as personalized speech synthesis, assistive technologies, and linguistic studies.

Please note:

- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal
  activities.
- Ensure compliance with local laws, regulations, and ethical standards when using this model.
- The developers assume no responsibility for any misuse of this model.

This project advocates the responsible development and use of artificial intelligence and encourages the community to
adhere to safety and ethical principles in AI research and applications.

## License and Acknowledgments

This project is built upon [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) and is distributed under the same
open-source license as SparkTTS. For details, please refer to the
original [SparkTTS License](https://github.com/SparkAudio/Spark-TTS/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HuiResearch/Fast-Spark-TTS&type=Date)](https://www.star-history.com/#HuiResearch/Fast-Spark-TTS&Date)