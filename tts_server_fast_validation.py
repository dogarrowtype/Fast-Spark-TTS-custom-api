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
import soxr
import demoji
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from semantic_text_splitter import TextSplitter
import whisper
from Levenshtein import ratio as levenshtein_ratio

# Local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fast_tts import AutoEngine

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global engine instance
engine = None

WHISPER_MODEL = None
VALIDATION_ENABLED = False
VALIDATION_THRESHOLD = 0.85  # Percentage of text that speech-to-text (whisper) says matches the input

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

def resample_audio_soxr(audio_data, input_sr, output_sr):
    """
    Resample audio using SoXR for highest quality resampling.
    
    Args:
        audio_data: Input audio as numpy array
        input_sr: Input sample rate
        output_sr: Output sample rate
    
    Returns:
        Resampled audio as numpy array
    """
    try:
        # SoXR resampling with highest quality settings
        resampled = soxr.resample(
            audio_data,
            input_sr,
            output_sr,
            quality='VHQ'  # Very High Quality - best available
        )
        logging.info(f"Resampled audio from {input_sr}Hz to {output_sr}Hz using SoXR VHQ")
        return resampled
    except Exception as e:
        logging.error(f"SoXR resampling failed: {e}, falling back to librosa")
        # Fallback to librosa if SoXR fails
        return librosa.resample(audio_data, orig_sr=input_sr, target_sr=output_sr, res_type='kaiser_best')

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

async def generate_audio_segment(engine, seg, prompt_speech_path, validate=False, validation_threshold=0.9):
    """Generate an audio segment asynchronously and optionally validate with Whisper STT"""
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

        # Load as numpy array first (so we always have the audio)
        y, sr = librosa.load(temp_path, sr=16000)

        # Validate with Whisper if enabled
        if validate and WHISPER_MODEL is not None:
            try:
                # Transcribe the audio
                logging.info(f"Validating audio segment with Whisper")
                result = WHISPER_MODEL.transcribe(temp_path, language="en")
                transcribed_text = result["text"].strip()

                # Compare with original text
                match_score = levenshtein_ratio(transcribed_text.lower(), seg.lower())

                logging.info(f"Validation score: {match_score:.2f}")
                logging.info(f"Original: {seg}")
                logging.info(f"Transcribed: {transcribed_text}")

                # Check if transcription ends with "..." (indicates cut-off audio)
                if transcribed_text.rstrip().endswith("..."):
                    logging.warning(f"Validation failed: transcription ends with '...' indicating cut-off audio")
                    return y, 0.0  # Return low score to trigger retry

                if match_score < validation_threshold:
                    logging.warning(f"Validation failed with score {match_score:.2f} < {validation_threshold}")
                    return y, match_score  # Return the audio anyway with the low score

                return y, match_score  # Return audio with actual score

            except Exception as e:
                logging.error(f"Error during whisper validation: {e}")
                # Continue without validation if an error occurs
                return y, 1.0

        return y, 1.0  # Return 1.0 as perfect score if validation was skipped or passed

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
    skip_model_init=False,
    validate=False,
    validation_threshold=0.9
):
    """
    Generates TTS audio from input text, splitting into segments if necessary.
    Now with optional Whisper validation.
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
    text = re.sub(r'—', ' — ', text)  # Em dash padding
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
        validation_success = False
        wav = None
        last_generated_wav = None
        last_score = 0.0
        
        while retry_count < MAX_RETRY_ATTEMPTS:
            try:
                # Generate audio asynchronously with validation
                result = await generate_audio_segment(
                    engine,
                    seg,
                    prompt_speech_path,
                    validate=validate,
                    validation_threshold=validation_threshold
                )

                if validate:
                    wav, score = result
                    # Store the last generated audio (now we always get audio)
                    last_generated_wav = wav
                    last_score = score
                    
                    if score < validation_threshold:  # Validation failed
                        logging.warning(f"Segment failed validation (score: {score:.2f}). Retry {retry_count+1}/{MAX_RETRY_ATTEMPTS}.")
                        retry_count += 1
                        
                        # If this was the last retry, break and use this attempt
                        if retry_count >= MAX_RETRY_ATTEMPTS:
                            logging.warning(f"Max validation retries reached. Using last generated audio (score: {score:.2f}).")
                            break
                        
                        # Vary the seed for different results and retry
                        seed = random.randint(0, 4294967295)
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        random.seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                        #await asyncio.sleep(0.5)  # Brief pause before retrying
                        continue
                    else:
                        validation_success = True
                        break  # Validation passed, use this audio
                else:
                    wav = result[0]  # Unpack when validate=False
                    validation_success = True
                    break  # No validation needed, use this audio

                # This code should not be reached due to breaks above, but keeping for safety
                break

            except Exception as e:
                logging.error(f"Error generating segment: {e}")
                retry_count += 1
                if retry_count >= MAX_RETRY_ATTEMPTS:
                    if validate and last_generated_wav is not None:
                        logging.warning(f"Max retries reached due to errors. Using last generated clip (validation score: {last_score:.2f}).")
                        wav = last_generated_wav
                    else:
                        logging.warning("Max retries reached. Could not generate audio.")
                        # Create a short silence as fallback
                        wav = np.zeros(int(16000 * 0.5))  # 0.5 seconds of silence
                    break
                #await asyncio.sleep(0.5)  # Brief pause before retrying

        # Now handle the silence trimming outside the retry loop
        if wav is not None:
            # Get both the trimmed audio and the amount of silence trimmed
            trimmed_wav, seconds_trimmed = trim_trailing_silence_librosa(wav, sample_rate=16000)

            # Always use the result at this point (no more retries)
            wavs.append(trimmed_wav)
            
            if seconds_trimmed > MAX_SILENCE_THRESHOLD:
                logging.warning(f"Final audio has {seconds_trimmed:.2f}s of silence (> {MAX_SILENCE_THRESHOLD}s). This usually means your clone clip is problematic.")
        else:
            logging.error("No audio generated for segment. Adding silence.")
            silence = np.zeros(int(16000 * 0.5))  # 0.5 seconds of silence
            wavs.append(silence)

        # Add a pause between segments
        delay_time = random.uniform(0.30, 0.5)  # Random delay between 300-500ms
        silence_samples = int(16000 * delay_time)  # 16000 is sample rate
        silence = np.zeros(silence_samples)
        wavs.append(silence)
        logging.info(f"Processed one segment{' after ' + str(retry_count) + ' retries' if retry_count > 0 else ''}.")

    final_wav = np.concatenate(wavs, axis=0)
    
    # Resample to 48kHz using SoXR for highest quality
    final_wav_48khz = resample_audio_soxr(final_wav, 16000, 48000)
    
    # Save the resampled audio at 48kHz
    sf.write(save_path, final_wav_48khz, samplerate=48000)
    logging.info(f"Audio saved at 48kHz: {save_path}")

    return save_path

async def convert_wav_to_mp3(wav_path):
    """
    Convert WAV file to MP3 format, silently failing to WAV if conversion is not possible.
    Now preserves 48kHz sample rate for high quality output.

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
            # Export with 48kHz sample rate and high bitrate
            sound.export(mp3_path, format="mp3", parameters=["-ar", "48000", "-b:a", "320k"])
            logging.info(f"MP3 exported at 48kHz with 320k bitrate")
            return mp3_path
        except ImportError:
            # Fall back to direct FFmpeg call if pydub is not available
            mp3_path = wav_path.replace('.wav', '.mp3')
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', wav_path, '-codec:a', 'libmp3lame',
                '-ar', '48000', '-b:a', '320k', mp3_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                logging.info(f"MP3 converted at 48kHz with 320k bitrate using FFmpeg")
                return mp3_path
            else:
                logging.warning("MP3 conversion failed. Returning WAV file instead.")
                return wav_path
    except Exception as e:
        logging.warning(f"Error converting to MP3: {e}. Returning WAV file instead.")
        return wav_path

async def convert_wav_to_opus(wav_path):
    """
    Convert WAV file to OPUS format at 128k bitrate, silently failing to WAV if conversion is not possible.
    Preserves 48kHz sample rate for high quality output.

    Args:
        wav_path: Path to the WAV file

    Returns:
        str: Path to the output file (either OPUS or original WAV if conversion failed)
    """
    if not ffmpeg_available:
        logging.warning("FFmpeg not available. Returning WAV file instead.")
        return wav_path

    try:
        # First try pydub approach
        try:
            from pydub import AudioSegment
            opus_path = wav_path.replace('.wav', '.opus')
            sound = AudioSegment.from_wav(wav_path)
            # Export with 48kHz sample rate and 128k bitrate for OPUS
            sound.export(opus_path, format="opus", parameters=["-ar", "48000", "-b:a", "128k"])
            logging.info(f"OPUS exported at 48kHz with 128k bitrate")
            return opus_path
        except ImportError:
            # Fall back to direct FFmpeg call if pydub is not available
            opus_path = wav_path.replace('.wav', '.opus')
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', wav_path, '-codec:a', 'libopus',
                '-ar', '48000', '-b:a', '128k', opus_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if os.path.exists(opus_path) and os.path.getsize(opus_path) > 0:
                logging.info(f"OPUS converted at 48kHz with 128k bitrate using FFmpeg")
                return opus_path
            else:
                logging.warning("OPUS conversion failed. Returning WAV file instead.")
                return wav_path
    except Exception as e:
        logging.warning(f"Error converting to OPUS: {e}. Returning WAV file instead.")
        return wav_path

async def convert_wav_to_flac(wav_path):
    """
    Convert WAV file to FLAC format with maximum compression, silently failing to WAV if conversion is not possible.
    Preserves 48kHz sample rate for high quality output.

    Args:
        wav_path: Path to the WAV file

    Returns:
        str: Path to the output file (either FLAC or original WAV if conversion failed)
    """
    if not ffmpeg_available:
        logging.warning("FFmpeg not available. Returning WAV file instead.")
        return wav_path

    try:
        # First try pydub approach
        try:
            from pydub import AudioSegment
            flac_path = wav_path.replace('.wav', '.flac')
            sound = AudioSegment.from_wav(wav_path)
            # Export with 48kHz sample rate and maximum compression for FLAC
            sound.export(flac_path, format="flac", parameters=["-ar", "48000", "-compression_level", "12"])
            logging.info(f"FLAC exported at 48kHz with maximum compression")
            return flac_path
        except ImportError:
            # Fall back to direct FFmpeg call if pydub is not available
            flac_path = wav_path.replace('.wav', '.flac')
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', wav_path, '-codec:a', 'flac',
                '-ar', '48000', '-compression_level', '12', flac_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if os.path.exists(flac_path) and os.path.getsize(flac_path) > 0:
                logging.info(f"FLAC converted at 48kHz with maximum compression using FFmpeg")
                return flac_path
            else:
                logging.warning("FLAC conversion failed. Returning WAV file instead.")
                return wav_path
    except Exception as e:
        logging.warning(f"Error converting to FLAC: {e}. Returning WAV file instead.")
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
        response_format = body.get('response_format', 'wav')

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
            segmentation_threshold=380,
            save_dir=temp_dir,
            validate=VALIDATION_ENABLED,
            validation_threshold=VALIDATION_THRESHOLD
        )

        # Convert to requested format (will silently fall back to WAV if FFmpeg isn't available)
        if response_format == 'mp3':
            output_file = await convert_wav_to_mp3(output_file)
        elif response_format == 'opus':
            output_file = await convert_wav_to_opus(output_file)
        elif response_format == 'flac':
            output_file = await convert_wav_to_flac(output_file)

        # Determine the correct mimetype based on the actual file extension
        if output_file.endswith('.mp3'):
            media_type = "audio/mpeg"
        elif output_file.endswith('.opus'):
            media_type = "audio/opus"
        elif output_file.endswith('.flac'):
            media_type = "audio/flac"
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
    parser.add_argument("--seg_threshold", type=int, default=380, help="Character limit for text segments")
    parser.add_argument("--allow_allcaps", action='store_true', help="Allow words with all capital letters")
    parser.add_argument("--validate", action='store_true', help="Enable validation of generated audio using Whisper")
    parser.add_argument("--validation_threshold", type=float, default=0.85, help="Threshold for text similarity validation (0-1)")
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

    # Determine appropriate device based on args
    if args.device == "default":
        # Determine appropriate device based on platform and availability
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            # macOS with MPS support (Apple Silicon)
            device = "mps"
            logging.info(f"Using MPS device")
        elif torch.cuda.is_available():
            # System with CUDA support
            device = "cuda"
            logging.info(f"Using CUDA device")
        else:
            # Fall back to CPU
            device = "cpu"
            logging.info("GPU acceleration not available, using CPU")
    else:
        # Verify CUDA availability if CUDA device is requested
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                logging.error("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
            else:
                device = args.device
                logging.info(f"Using CUDA device: {device}")
        else:
            device = args.device
            logging.info(f"Using device: {device}")

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
                    llm_device=device,
                    tokenizer_device=device,
                    detokenizer_device=device,
                )
                logging.info("Model loaded successfully")
            except Exception as e:
                logging.error(f"Error preloading model: {e}")
                sys.exit(1)

    ffmpeg_available = is_ffmpeg_available()
    logging.info(f"FFmpeg is {'available' if ffmpeg_available else 'not available'}")
    # Preload Whisper model if validation enabled
    if args.validate and ffmpeg_available:
        try:
            logging.info("Loading Whisper tiny model for validation...")
            VALIDATION_ENABLED = True
            VALIDATION_THRESHOLD = args.validation_threshold
            WHISPER_MODEL = whisper.load_model("tiny")
            logging.info("Whisper model loaded successfully")
        except Exception as e:
            logging.error(f"Error setting up Whisper: {e}. Validation will be disabled.")
            VALIDATION_ENABLED = False


    # Run startup tasks
    asyncio.run(startup_load_model())

    # Start the server
    logging.info(f"Starting FastAPI TTS server on http://{args.host}:{args.port}/v1/audio/speech")
    uvicorn.run(app, host=args.host, port=args.port)