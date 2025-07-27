import sys
import os
import shutil
import subprocess
import platform
import logging
import tempfile
import random
import json
import re
import uuid
import time
import asyncio
from datetime import datetime
from typing import Optional

# Third-party libraries
import torch
import numpy as np
import soundfile as sf
import librosa
import demoji
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from semantic_text_splitter import TextSplitter

# Local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fast_tts import AutoEngine

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global engine instance
engine = None

# Global configuration for pipe splitting
PIPE_SPLIT_ENABLED = True  # Enable pipe character splitting by default

# Global voice parameters set at server startup
GLOBAL_VOICE_PARAMS = {
    "gender": None,
    "pitch": None,
    "speed": None,
    "emotion": None,
    "seed": None,
    "prompt_speech_path": None,
    "prompt_text": None,
    "max_generation_tokens": 1500,
}

app = FastAPI(title="Fast-Spark-TTS Server",
              description="OpenAI-compatible TTS API for SparkTTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Model for TTS request
class TTSRequest(BaseModel):
    model: str = "tts-1"
    input: str
    response_format: str = "mp3"

def is_ffmpeg_available():
    """Check if ffmpeg is available on the system path, works on Windows and Linux."""
    # Check if we're on Windows
    is_windows = platform.system().lower() == "windows"
    # On Windows, we should also check for ffmpeg.exe specifically
    ffmpeg_commands = ['ffmpeg.exe', 'ffmpeg'] if is_windows else ['ffmpeg']

    # Method 1: Using shutil.which (works on both platforms)
    for cmd in ffmpeg_commands:
        if shutil.which(cmd) is not None:
            return True

    # Method 2: Fallback to subprocess
    for cmd in ffmpeg_commands:
        try:
            # On Windows, shell=True might be needed in some environments
            subprocess.run(
                [cmd, '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=5,
                shell=is_windows
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            continue

    return False

def split_on_pipes(text):
    """
    Split text on pipe characters (|) while preserving natural pauses.
    Returns a list of text segments.
    """
    # Split on pipe character
    segments = text.split('|')
    
    # Clean up segments - strip whitespace and filter out empty ones
    cleaned_segments = []
    for segment in segments:
        cleaned = segment.strip()
        if cleaned:  # Only add non-empty segments
            cleaned_segments.append(cleaned)
    
    return cleaned_segments

ffmpeg_available = is_ffmpeg_available()
logging.info(f"FFmpeg is {'available' if ffmpeg_available else 'not available'}")

async def generate_audio_segment(engine, seg, prompt_speech_path):
    """Generate an audio segment asynchronously"""
    # Create a temporary file with a .wav extension
    # Have to do temporary .wav because engine.write_audio is hardcoded for file output
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        clip = await engine.clone_voice_async(
            text=seg,
            reference_audio=prompt_speech_path,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            max_tokens=GLOBAL_VOICE_PARAMS["max_generation_tokens"],
            repetition_penalty=1.0,
            length_threshold=GLOBAL_VOICE_PARAMS["max_generation_tokens"],
            window_size=GLOBAL_VOICE_PARAMS["max_generation_tokens"],
        )

        # Write to the temporary file
        engine.write_audio(clip, temp_path)

        # Load as numpy array
        y, sr = librosa.load(temp_path, sr=16000)
        return y
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

async def generate_tts_audio(
    text,
    model_dir=None,
    device="cuda:0",
    prompt_speech_path=None,
    prompt_text=None,
    gender=None,
    pitch=None,
    speed=None,
    emotion=None,
    save_dir="example/results",
    segmentation_threshold=None,
    seed=None,
    model=None,
    skip_model_init=False
):
    """
    Generates TTS audio from input text, splitting into segments if necessary.
    Now fully asynchronous.
    """
    if model_dir is None:
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "Spark-TTS-0.5B"))

    global engine

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Seed set to: {seed}")
    else:
        seed = random.randint(0, 4294967295)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Seed set to: {seed}")

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    # Text preprocessing (standardize quotes, etc.)
    text = re.sub(r'[""„‟«»❝❞〝〞〟＂""＂]', '"', text)  # Convert fancy double quotes
    text = re.sub(r'[''‚‛‹›❛❜`´'']', "'", text)  # Convert fancy single quotes/apostrophes
    text = re.sub(r'[–]', '-', text)  # En dash to hyphen
    text = re.sub(r'…', '...', text)  # Ellipsis
    text = re.sub(r'[•‣⁃*]', '', text)  # Bullets to none
    text = demoji.replace(text, "")

    if not args.allow_allcaps:
        # Convert all-caps words to lowercase (model chokes on all caps)
        def lowercase_all_caps(match):
            word = match.group(0)
            if word.isupper() and len(word) > 1:
                return word.lower()
            return word
        text = re.sub(r'\b[A-Z][A-Z]+\b', lowercase_all_caps, text)

    # Handle pipe character splitting first (if enabled)
    if PIPE_SPLIT_ENABLED:
        pipe_segments = split_on_pipes(text)
        logging.info(f"Text split on pipe characters into {len(pipe_segments)} parts")
    else:
        pipe_segments = [text]

    # Apply semantic segmentation to each pipe segment
    all_segments = []
    for pipe_seg in pipe_segments:
        if pipe_seg.strip():  # Skip empty segments
            logging.info(f"Splitting pipe segment into semantic segments... Threshold: {segmentation_threshold} characters")
            splitter = TextSplitter(segmentation_threshold)
            semantic_segments = splitter.chunks(pipe_seg.strip())
            all_segments.extend(semantic_segments)
    
    segments = all_segments
    logging.info(f"Total number of segments after pipe and semantic splitting: {len(segments)}")

    MAX_SILENCE_THRESHOLD = 5.0  # 5 seconds
    MAX_RETRY_ATTEMPTS = 3

    wavs = []
    for seg in segments:
        logging.info(f"Processing one segment:\n{seg}")
        retry_count = 0
        while retry_count < MAX_RETRY_ATTEMPTS:
            try:
                # Generate audio asynchronously
                wav = await generate_audio_segment(engine, seg, prompt_speech_path)

                # Get both the trimmed audio and the amount of silence trimmed
                trimmed_wav, seconds_trimmed = trim_trailing_silence_librosa(wav, sample_rate=16000)

                # If silence is acceptable, or we've tried too many times, use this result
                if seconds_trimmed < MAX_SILENCE_THRESHOLD or retry_count == MAX_RETRY_ATTEMPTS - 1:
                    wavs.append(trimmed_wav)
                    break
                else:
                    logging.warning(f"Too much silence detected ({seconds_trimmed:.2f}s > {MAX_SILENCE_THRESHOLD}s). "
                                "Retrying segment... (This usually means your clone clip is bad.)")
                    retry_count += 1
                    # Vary the seed for different results
                    seed = random.randint(0, 4294967295)
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
            except Exception as e:
                logging.error(f"Error generating segment: {e}")
                retry_count += 1
                if retry_count >= MAX_RETRY_ATTEMPTS:
                    logging.warning("Max retries reached. Adding silence instead.")
                    silence = np.zeros(int(16000 * 0.5))  # 0.5 seconds of silence
                    wavs.append(silence)
                    break
                await asyncio.sleep(0.5)  # Brief pause before retrying

        # Add a pause between segments
        delay_time = random.uniform(0.30, 0.5)  # Random delay between 300-500ms
        silence_samples = int(16000 * delay_time)  # 16000 is sample rate
        silence = np.zeros(silence_samples)
        wavs.append(silence)

        logging.info(f"Processed one segment{' after ' + str(retry_count) + ' retries' if retry_count > 0 else ''}.")

    final_wav = np.concatenate(wavs, axis=0)
    sf.write(save_path, final_wav, samplerate=16000)
    logging.info(f"Audio saved at: {save_path}")

    return save_path

async def convert_wav_to_mp3(wav_path):
    """
    Convert WAV file to MP3 format, silently failing to WAV if conversion is not possible

    Args:
        wav_path: Path to the WAV file

    Returns:
        str: Path to the output file (either MP3 or original WAV if conversion failed)
    """
    if not ffmpeg_available:
        logging.warning("FFmpeg not available. Returning WAV file instead.")
        return wav_path

    try:
        # First try pydub approach
        try:
            from pydub import AudioSegment
            mp3_path = wav_path.replace('.wav', '.mp3')
            sound = AudioSegment.from_wav(wav_path)
            sound.export(mp3_path, format="mp3", parameters=["-q:a", "0"])
            return mp3_path
        except ImportError:
            # Fall back to direct FFmpeg call if pydub is not available
            mp3_path = wav_path.replace('.wav', '.mp3')
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', wav_path, '-codec:a', 'libmp3lame',
                '-qscale:a', '2', mp3_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                return mp3_path
            else:
                logging.warning("MP3 conversion failed. Returning WAV file instead.")
                return wav_path
    except Exception as e:
        logging.warning(f"Error converting to MP3: {e}. Returning WAV file instead.")
        return wav_path

def trim_trailing_silence_librosa(wav_data, sample_rate=16000, top_db=30, frame_length=1024, hop_length=512):
    """
    Trims trailing silence using librosa's effects.trim and returns both trimmed audio and seconds trimmed
    """
    try:
        # Trim starting and trailing silence
        y_trimmed, _ = librosa.effects.trim(
            wav_data,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        seconds_trimmed = (len(wav_data) - len(y_trimmed)) / sample_rate
        logging.info(f"Trimmed {seconds_trimmed:.2f}s of silence using librosa")
        return y_trimmed, seconds_trimmed
    except Exception as e:
        logging.warning(f"Error trimming silence with librosa: {e}")
        return wav_data, 0.0

# OpenAI compatible TTS endpoint
@app.post('/v1/audio/speech')
async def tts(request: Request):
    try:
        # Parse request body manually to handle different content types
        content_type = request.headers.get('content-type', '')

        if 'application/json' in content_type:
            # Handle JSON
            body = await request.json()
        elif 'application/x-www-form-urlencoded' in content_type:
            # Handle form data
            form_data = await request.form()
            body = dict(form_data)
        else:
            # Default fallback
            body = await request.json()

        # Log the incoming request
        client_ip = request.client.host

        # Print request details to console
        print("\n==== TTS REQUEST ====")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Client IP: {client_ip}")
        print(f"Content-Type: {content_type}")
        print("Request data:")
        print(json.dumps(body, indent=2))
        print("====================\n")

        # Extract parameters similar to OpenAI's API
        model = body.get('model', 'tts-1')  # Default to tts-1
        input_text = body.get('input')
        response_format = body.get('response_format', 'mp3')

        if not input_text:
            raise HTTPException(status_code=400, detail="Input text is required")

        # Create a temp directory for outputs
        temp_dir = tempfile.mkdtemp()

        # Generate the audio using global voice parameters
        output_file = await generate_tts_audio(
            text=input_text,
            gender=GLOBAL_VOICE_PARAMS["gender"],
            pitch=GLOBAL_VOICE_PARAMS["pitch"],
            speed=GLOBAL_VOICE_PARAMS["speed"],
            emotion=GLOBAL_VOICE_PARAMS["emotion"],
            seed=GLOBAL_VOICE_PARAMS["seed"],
            prompt_speech_path=GLOBAL_VOICE_PARAMS["prompt_speech_path"],
            prompt_text=GLOBAL_VOICE_PARAMS["prompt_text"],
            segmentation_threshold=400,
            save_dir=temp_dir
        )

        # Convert to MP3 if requested (will silently fall back to WAV if FFmpeg isn't available)
        if response_format == 'mp3':
            output_file = await convert_wav_to_mp3(output_file)

        # Determine the correct mimetype based on the actual file extension
        if output_file.endswith('.mp3'):
            media_type = "audio/mpeg"
        else:
            media_type = "audio/wav"

        # Return the audio file
        return FileResponse(path=output_file, media_type=media_type, filename=os.path.basename(output_file))

    except Exception as e:
        logging.error(f"Error in TTS endpoint: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.options('/v1/audio/speech')
async def tts_options():
    # This handles OPTIONS requests explicitly
    return {}

@app.get('/v1/models')
async def list_models():
    """Mimics OpenAI's models endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai",
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai",
            }
        ]
    }

# Health check endpoint
@app.get('/health')
async def health_check():
    voice_info = {
        "gender": GLOBAL_VOICE_PARAMS["gender"],
        "pitch": GLOBAL_VOICE_PARAMS["pitch"],
        "speed": GLOBAL_VOICE_PARAMS["speed"],
        "emotion": GLOBAL_VOICE_PARAMS["emotion"],
        "using_prompt": GLOBAL_VOICE_PARAMS["prompt_speech_path"] is not None
    }
    return {
        "status": "ok",
        "message": "SparkTTS server is running",
        "voice_config": voice_info
    }

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='OpenAI-compatible TTS Server for SparkTTS')
    parser.add_argument('--port', type=int, default=9991, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--device', type=str, default='default', help='Device to use for model inference')
    parser.add_argument('--model_dir', type=str, default="models/Spark-TTS-0.5B", help='Path to the SparkTTS model')

    # Voice configuration arguments
    parser.add_argument('--prompt_audio', type=str, help='Path to audio file for voice cloning')
    parser.add_argument('--prompt_text', type=str, help='Transcript text for the prompt audio (optional)')
    parser.add_argument('--gender', type=str, choices=["male", "female"], help='Gender parameter')
    parser.add_argument('--pitch', type=str, choices=["very_low", "low", "moderate", "high", "very_high"],
                        default="moderate", help='Pitch parameter')
    parser.add_argument('--speed', type=str, choices=["very_low", "low", "moderate", "high", "very_high"],
                        default="moderate", help='Speed parameter')
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible voice generation")
    parser.add_argument("--seg_threshold", type=int, default=400, help="Character limit for text segments")
    parser.add_argument("--allow_allcaps", action='store_true', help="Allow words with all capital letters")
    parser.add_argument("--disable_pipe_split", action='store_true', help="Disable automatic text splitting on pipe characters (|)")

    args = parser.parse_args()

    # Set global pipe split configuration
    PIPE_SPLIT_ENABLED = not args.disable_pipe_split

    # Set global voice parameters
    GLOBAL_VOICE_PARAMS["gender"] = args.gender if not args.prompt_audio else None
    GLOBAL_VOICE_PARAMS["pitch"] = args.pitch if not args.prompt_audio else None
    GLOBAL_VOICE_PARAMS["speed"] = args.speed if not args.prompt_audio else None
    GLOBAL_VOICE_PARAMS["seed"] = args.seed
    GLOBAL_VOICE_PARAMS["prompt_speech_path"] = os.path.abspath(args.prompt_audio) if args.prompt_audio else None
    GLOBAL_VOICE_PARAMS["prompt_text"] = args.prompt_text

    # Log voice configuration
    if args.prompt_audio:
        # Normalize path + validate
        args.prompt_audio = os.path.abspath(args.prompt_audio)
        if not os.path.exists(args.prompt_audio):
            logging.error(f"❌ Prompt audio file not found: {args.prompt_audio}")
            sys.exit(1)

        # Log cloning info
        logging.info("🔊 Voice cloning mode enabled")
        logging.info(f"🎧 Cloning from: {args.prompt_audio}")

        # Log audio info
        try:
            info = sf.info(args.prompt_audio)
            logging.info(f"📏 Prompt duration: {info.duration:.2f} seconds | Sample Rate: {info.samplerate}")
        except Exception as e:
            logging.warning(f"⚠️ Could not read prompt audio info: {e}")

        # Override pitch/speed/gender when using voice cloning
        if args.gender or args.pitch or args.speed:
            print("[!] Warning: Voice cloning mode detected — ignoring gender/pitch/speed settings.")
        args.gender = None
        args.pitch = None
        args.speed = None
    else:
        logging.info(f"🔊 Using configured voice: Gender={args.gender}, Pitch={args.pitch}, Speed={args.speed}")
        if args.seed:
            logging.info(f"🎲 Fixed seed: {args.seed}")

    # Preload the model on startup if model_dir is provided
    async def startup_load_model():
        global engine
        if args.model_dir:
            try:
                logging.info(f"Preloading SparkTTS model from {args.model_dir}")
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                engine = AutoEngine(
                    model_path=args.model_dir,
                    max_length=GLOBAL_VOICE_PARAMS["max_generation_tokens"],
                    backend="llama-cpp",
                )
                logging.info("Model loaded successfully")
            except Exception as e:
                logging.error(f"Error preloading model: {e}")
                sys.exit(1)

    # Run startup tasks
    asyncio.run(startup_load_model())

    # Start the server
    logging.info(f"Starting FastAPI TTS server on http://{args.host}:{args.port}/v1/audio/speech")
    uvicorn.run(app, host=args.host, port=args.port)