# Fast-Spark-TTS Custom API

An OpenAI-compatible Text-to-Speech API server for SparkTTS with advanced voice cloning, validation, and intelligent text processing capabilities. It works on both CUDA gpus and Apple Silicon!

The main goal: send text via a widely used api and get clean, usable audio almost every time, from a model with issues.

Fully functional, but may encounter some known issues that are being worked on.

## Overview

This project provides a production-ready OpenAI-compatible TTS API built on top of SparkTTS. It transforms the original SparkTTS into a seamless drop-in replacement for OpenAI's TTS service, making it compatible with applications like [SillyTavern](https://github.com/SillyTavern/SillyTavern) and other OpenAI TTS clients.

### Key Features

- **OpenAI Compatibility**: Full compatibility with OpenAI's `/v1/audio/speech` API endpoint
- **Advanced Voice Cloning**: Zero-shot voice cloning using reference audio samples (5-20 seconds)
- **Intelligent Text Processing**: Semantic text splitting to handle unlimited text length
- **Pipe Character Splitting**: Automatic segment splitting on pipe characters (`|`) for precise audio control
- **Audio Validation**: Optional Whisper-based validation to ensure quality output
- **Multiple Backends**: Support for llama-cpp for inference
- **Smart Retry Logic**: Automatic retry for failed segments with quality control
- **Flexible Audio Formats**: Support for MP3, WAV, OPUS, FLAC, AAC, and PCM formats
- **Production Features**: Silence trimming, semantic segmentation, and robust error handling

## Hardware Requirements

- **VRAM**: Not much. Depends on which quant you choose. Less than 4GB.
- **Performance**: Larger GPUs will achieve faster than real-time generation. The longer the input segment is, the longer it takes to generate. No streaming, sorry. On a high end mac, it can be extremely fast. (3 seconds for 10 seconds of audio, but it varies.)
- **CPU**: GPU recommended, but can run on CPU only.

## Installation

### Prerequisites

- Python 3.11 or 3.12
- CUDA-compatible GPU (recommended), Apple Silicon with MPS support, GPU that supports vulkan, or a fast CPU
- Git (for cloning)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/Fast-Spark-TTS-custom-api
cd Fast-Spark-TTS-custom-api
```

### Step 2: Environment Setup

**Option A: Conda (Recommended)**
```bash
conda create -n sparktts python=3.12 -y
conda activate sparktts
pip install -r requirements.txt
```

**Option B: Virtual Environment**
```bash
python3.12 -m venv ./venv
source ./venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Install Dependencies

First, install the core dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: System-Specific llama-cpp-python Installation

The `llama-cpp-python` package must be installed separately based on your system configuration for optimal performance. Choose the appropriate installation method:

#### For macOS (Apple Silicon/Intel)

```bash
pip install llama-cpp-python
```

#### For CUDA GPU Systems (For NVIDIA GPUs)

**Pre-built Wheels (Easiest Method)**

For systems with CUDA >=12.1 and Python 3.10-3.12:

```bash
# Replace <cuda-version> with your CUDA version
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version-goes-here>
```

**CUDA Version Mapping:**
- `cu121`: CUDA 12.1
- `cu122`: CUDA 12.2
- `cu123`: CUDA 12.3
- `cu124`: CUDA 12.4
- `cu125`: CUDA 12.5

**Example for CUDA 12.1:**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```
Install the highest version that your system supports.

**Manual CUDA Installation (Alternative)**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

#### For CPU-Only Systems (install openblas)

```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

#### For AMD GPUs (ROCm)

```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python
```

#### For Vulkan (generic gpu)

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python
```

#### For Other Backends

For additional backends (OpenCL, etc.), see the comprehensive guide: [https://github.com/abetlen/llama-cpp-python#supported-backends](https://github.com/abetlen/llama-cpp-python#supported-backends)

**Installation Verification:**

After installation, verify it works correctly:

```python
python -c "from llama_cpp import Llama; print('llama-cpp-python installed successfully')"
```

### Step 5: Download Model

**Option A: Using huggingface-cli (Recommended)**

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Download model files to the models directory (excluding model.safetensors)
huggingface-cli download SparkAudio/Spark-TTS-0.5B --local-dir "models/Spark-TTS-0.5B" --exclude "LLM/model.safetensors"

# Download the GGUF model file and rename it
huggingface-cli download mradermacher/SparkTTS-LLM-GGUF --filename "SparkTTS-LLM.f16.gguf" --local-dir-use-symlinks False --local-dir "models/temp"
mv models/temp/SparkTTS-LLM.f16.gguf models/Spark-TTS-0.5B/LLM/model.gguf
rmdir models/temp
```

**Option B: Using wget**

```bash
# Create the models directory structure
mkdir -p models/Spark-TTS-0.5B/LLM

# Download the GGUF model file
wget https://huggingface.co/mradermacher/SparkTTS-LLM-GGUF/resolve/main/SparkTTS-LLM.f16.gguf -O models/Spark-TTS-0.5B/LLM/model.gguf

# Download all other files from the main repository (excluding model.safetensors)
pip install huggingface-hub
huggingface-cli download SparkAudio/Spark-TTS-0.5B --local-dir "models/Spark-TTS-0.5B" --exclude "LLM/model.safetensors"
```

**Important Notes:**
- This program requires the `.gguf` version of the model, not the `.safetensors` version
- The original `model.safetensors` file from the SparkAudio repository must be excluded
- All other files from the SparkAudio/Spark-TTS-0.5B repository are required
- The downloaded `.gguf` file must be renamed to `model.gguf` in the `LLM` directory
- F16 is good, but smaller versions will work if your application requires a smaller footprint

### Step 6: Optional Dependencies

**FFmpeg (Recommended for MP3 support)**
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

**For Validation Features**
The server includes Whisper-based validation. Dependencies are included in `requirements.txt`.

## Quick Start

### Server with Validation (Recommended)

Enable Whisper validation for quality assurance (will download whisper-tiny model):

```bash
python tts_server_fast_validation.py --prompt_audio "voice.wav" --validate --validation_threshold 0.85
```

### Basic Server (Voice Cloning)

Start the server with a reference voice for cloning:

```bash
python tts_server_fast_validation.py --prompt_audio "path/to/your/voice.wav"
```

### Server with Configured Voice

Start with specific voice parameters (no cloning):

```bash
python tts_server_fast_validation.py --gender male --pitch moderate --speed moderate
```


## API Usage

### OpenAI Compatible Endpoint

The server provides a standard OpenAI-compatible endpoint:

**Endpoint**: `POST http://localhost:9991/v1/audio/speech`

**Request Format**:
```json
{
    "model": "tts-1",
    "input": "Hello, this is a test of the speech synthesis system.",
    "response_format": "mp3"
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:9991/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello world!",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:9991/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Hello, this is SparkTTS!",
        "response_format": "mp3"
    }
)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

### Using with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9991/v1",
    api_key="not-needed"
)

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="default",
    input="Hello from SparkTTS!"
) as response:
    response.stream_to_file("speech.mp3")
```

## Command Line Options

### Server Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Host to run the server on | `127.0.0.1` |
| `--port` | Port to run the server on | `9991` |
| `--device` | Device for model inference (`default`, `cuda:0`, `cpu`) | `default` |
| `--model_dir` | Path to the SparkTTS model directory | `models/Spark-TTS-0.5B` |

### Voice Configuration

| Option | Description | Choices |
|--------|-------------|---------|
| `--prompt_audio` | Path to audio file for voice cloning | - |
| `--prompt_text` | Transcript text for prompt audio (optional) | - |
| `--gender` | Gender parameter (when not using cloning) | `male`, `female` |
| `--pitch` | Pitch level | `very_low`, `low`, `moderate`, `high`, `very_high` |
| `--speed` | Speed level | `very_low`, `low`, `moderate`, `high`, `very_high` |
| `--seed` | Random seed for reproducible generation | - |

### Text Processing

| Option | Description | Default |
|--------|-------------|---------|
| `--seg_threshold` | Character limit for text segments | `380` |
| `--allow_allcaps` | Allow words with all capital letters | `False` |
| `--disable_pipe_split` | Disable automatic text splitting on pipe characters | `False` |

### Validation Features

| Option | Description | Default |
|--------|-------------|---------|
| `--validate` | Enable Whisper validation of generated audio | `False` |
| `--validation_threshold` | Similarity threshold for validation (0-1) | `0.85` |

## Text Processing Features

### Pipe Character Splitting

The server includes automatic text splitting on pipe characters (`|`) for precise control over audio segments. This feature is **enabled by default** and allows you to create natural pauses or breaks in the generated speech.

**How it works:**
- Text is first split on pipe characters (`|`)
- Each pipe-separated segment is then processed through semantic segmentation
- This creates natural breaks in the audio at the pipe locations

**Example Usage:**
```json
{
  "model": "tts-1",
  "input": "Welcome to our service. | Please hold while we connect you. | Thank you for your patience.",
  "response_format": "mp3"
}
```

This will generate three distinct audio segments with natural pauses between them.

**Disabling Pipe Splitting:**
If you want to treat pipe characters as literal text instead of segment breaks, use the `--disable_pipe_split` flag:

```bash
python tts_server_fast_validation.py --prompt_audio "voice.wav" --disable_pipe_split
```

**Use Cases:**
- Creating distinct sections in narration
- Adding natural pauses in dialogue
- Separating different speakers or contexts
- Building interactive voice responses with clear breaks

## Voice Cloning

### Creating Reference Audio

For optimal voice cloning results:

1. **Audio Requirements**:
   - 5-20 seconds duration
   - Clear, noise-free audio
   - Single speaker, continuous speech
   - WAV format recommended (other formats may work)

2. **Best Practices**:
   - Use a complete sentence or paragraph
   - Avoid background music or noise
   - Ensure consistent volume
   - Record in good quality (16kHz+ sample rate)

### Voice Sample Examples

The following voice samples demonstrate the quality and variety achievable with SparkTTS. These can serve as reference for the types of voices you can clone or generate.

These files are included, and are ready to be used to generate. To use them, input the file after `--prompt_audio`

**Provided Voice Descriptions**:
- **female1.wav**: melancholy, rainy
- **female2.wav**: confident, expressive
- **female3.wav**: anime VA, cutesy, kinda sad, amazed
- **female4.wav**: clean, light, a touch of mystery
- **male1.wav**: cute, super expressive, adventurous
- **male2.wav**: determined, focused
- **male3.wav**: surprised, best for short bursts
- **male4.wav**: deep, rich, noir, serious
- **male5.wav**: alternate version of male4.wav, expressive, deep, rich, engaging

---

<table>
<tr>
<td align="center">
    
**Female 1**
</td>
<td align="center">
    
**Female 2**
</td>
</tr>

<tr>
<td align="center">

[female1.webm](https://github.com/user-attachments/assets/f859e295-a290-49be-8c69-f793bf69e3e0)

</td>
<td align="center">
    
[female2.webm](https://github.com/user-attachments/assets/77488785-5d48-477b-a2d4-33db4abe95fd)

</td>
</tr>
</table>

---

<table>
<tr>
<td align="center">
    
**Female 3**
</td>
<td align="center">
    
**Female 4**
</td>
</tr>

<tr>
<td align="center">

[female3.webm](https://github.com/user-attachments/assets/d84cb42e-6a1c-49de-a397-606e0fafbd82)

</td>
<td align="center">
    
[female4.webm](https://github.com/user-attachments/assets/1c931e30-8669-4f55-addd-e21ab94687e5)

</td>
</tr>
</table>

---

<table>
<tr>
<td align="center">
    
**Male 1**
</td>
<td align="center">
    
**Male 2**
</td>
</tr>

<tr>
<td align="center">

[male1.webm](https://github.com/user-attachments/assets/c29ba86c-9394-4b28-bcf6-1b24c02dbf02)

</td>
<td align="center">
    
[male2.webm](https://github.com/user-attachments/assets/d79030ec-ffdc-437f-8333-0b6010e6d374)

</td>
</tr>
</table>

---

<table>
<tr>
<td align="center">
    
**Male 3**
</td>
<td align="center">
    
**Male 4**
</td>
</tr>

<tr>
<td align="center">

[male3.webm](https://github.com/user-attachments/assets/3bc720d6-286e-4904-8d57-fdea3055c288)

</td>
<td align="center">
    
[male4.webm](https://github.com/user-attachments/assets/da9ff132-3508-46c0-ba4c-1c0658a0624d)

</td>
</tr>
</table>

---

<table>
<tr>
<td align="center">
    
**Male 5**
</td>
<td align="center">
    
**Additional Voices**
</td>
</tr>

<tr>
<td align="center">

[male5.webm](https://github.com/user-attachments/assets/ad830d60-a493-4736-a7da-49480c3d0190)

</td>
<td align="center">

More voices can be created through cloning

</td>
</tr>
</table>

---

### Second-Generation Cloning

For even better results, use a two-step process:

1. Generate audio with your reference voice
2. Select the best result that captures the desired accent/tone
3. Use that generated audio as the new reference for future generations

**Example phrase for testing**:
```
In these days of skeptical enlightened civilization one cannot claim to have seen a sea serpent and expect anything but laughter. And this was even more incredible.
```

## Integration Examples

### SillyTavern Setup

1. In SillyTavern, go to Extensions → TTS
2. Set TTS Provider to "OpenAI"
3. Set API URL to: `http://localhost:9991/v1`
4. Enable "Narrate by paragraphs (when not streaming)" for better performance

### Open-WebUI Setup

1. In Open-WebUI admin panel, go to "Audio"
2. Set TTS provider to: `http://localhost:9991/v1`
3. Set "response splitting" to "paragraphs"

## Advanced Configuration

### Multiple Server Instances

Run different instances for different voices:

```bash
# Female voice server on port 9991
python tts_server_fast_validation.py --prompt_audio "female_voice.wav" --port 9991

# Male voice server on port 9992  
python tts_server_fast_validation.py --prompt_audio "male_voice.wav" --port 9992
```

### Performance Tuning

- **Reduce segment threshold** for faster initial response: `--seg_threshold 200`
- **Enable validation** for quality assurance: `--validate --validation_threshold 0.85`
- **Use fixed seed** for consistent results: `--seed 42`

## Troubleshooting

### Known issues
- Fails on reading numbers expanded to text.
- Has issues reading complex formatting (lists, typical assistant output). Mostly suitable for reading prose, conversations, natural speech.
- Uncommon names have inconsistent readings.

### Common Issues

1. **Model Not Found**: Ensure the model is downloaded to `models/Spark-TTS-0.5B`
2. **Audio Quality Issues (Skipping text)**: Enable validation with `--validate`
2. **Slow Performance**: Check GPU utilization and consider reducing `seg_threshold`

### Validation Failures

If validation is enabled and segments are being retried:
- Check reference audio quality
- Adjust `--validation_threshold` (lower = more permissive, more prone to making mistakes).
- Above 0.95 is too high, as it will often toss out completely fine gens as being too imperfect.
- Ensure the reference voice matches the target language

### Audio Format Issues

- MP3 conversion requires FFmpeg
- If FFmpeg is unavailable, the server will fall back to WAV format
- PCM format returns raw audio data without headers

## Performance Notes

- **Concurrent Requests**: Supported but not as stable as single-user mode
- **Streaming**: Not yet implemented (generates complete audio before sending)
- **Memory Usage**: ~8.5 GB VRAM for the model
- **Speed**: ~2x real-time on RTX 3060, faster on more powerful GPUs

## License and Acknowledgments

This project is built upon [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) and maintains compatibility with the original SparkTTS license. Special thanks to [@AcTePuKc](https://github.com/AcTePuKc) and [HuiResearch/Fast-Spark-TTS](https://github.com/HuiResearch/Fast-Spark-TTS) for foundational work.

### Usage Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and legitimate applications such as personalized speech synthesis, assistive technologies, and language research.

**Please note:**
- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deep fakes, or any other illegal activities
- Ensure that you comply with local laws and ethical standards when using this model
- The developers are not responsible for any misuse of this model

This project advocates responsible AI development and use, and encourages the community to uphold safety and ethical principles in AI research and applications.

## Related Projects

- **Original SparkTTS**: [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
- **FastTTS Platform**: [HuiResearch/Fast-Spark-TTS](https://github.com/HuiResearch/Fast-Spark-TTS)
- **Quantized Models**: [mradermacher/SparkTTS-LLM-GGUF](https://huggingface.co/mradermacher/SparkTTS-LLM-GGUF)