import os
import math
import wave
import contextlib
import tempfile
import subprocess
import shutil
from typing import List


def get_audio_duration(path: str) -> float:
    """Return duration of WAV file in seconds."""
    with contextlib.closing(wave.open(path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def split_wav_into_chunks(path: str, chunk_length: int = 300) -> List[str]:
    """Split *path* into ``chunk_length`` second segments using ffmpeg.

    Returns list of generated file paths.
    """
    duration = get_audio_duration(path)
    num_chunks = int(math.ceil(duration / chunk_length))

    temp_dir = tempfile.mkdtemp(prefix="wav_chunks_")
    chunk_paths = []

    # Verify that ffmpeg is installed and accessible
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it is in your PATH.")

    for idx in range(num_chunks):
        start = idx * chunk_length
        output = os.path.join(temp_dir, f"chunk_{idx:04d}.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            path,
            "-ss",
            str(start),
            "-t",
            str(chunk_length),
            output,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='ignore')}") from e
        chunk_paths.append(output)
    return chunk_paths


def transcribe_chunks(paths: List[str], use_faster: bool = False) -> str:
    """Transcribe each WAV file listed in *paths* and return the concatenated text."""
    if use_faster:
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("base", device=device, compute_type=compute_type)
        texts = []
        for idx, p in enumerate(paths, 1):
            print(f"Transcribing chunk {idx}/{len(paths)}...")
            segments, _ = model.transcribe(p, language="fr", beam_size=5)
            chunk_text = "".join(seg.text for seg in segments)
            texts.append(chunk_text.strip())
    else:
        import whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", device=device)
        texts = []
        for idx, p in enumerate(paths, 1):
            print(f"Transcribing chunk {idx}/{len(paths)}...")
            result = model.transcribe(p, language="fr", fp16=True, beam_size=5)
            chunk_text = "".join(seg["text"] for seg in result.get("segments", []))
            texts.append(chunk_text.strip())
    return "\n".join(texts)


def transcribe_wav_in_chunks(path: str, output_path: str = "transcript.txt", chunk_length: int = 300, use_faster: bool = False) -> str:
    """Split *path* into chunks, transcribe them and write the final transcript.

    Returns path to the transcript.
    """
    chunk_paths: List[str] = []
    try:
        chunk_paths = split_wav_into_chunks(path, chunk_length=chunk_length)
        text = transcribe_chunks(chunk_paths, use_faster=use_faster)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
    finally:
        for p in chunk_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        # remove temporary directory
        temp_dir = os.path.dirname(chunk_paths[0]) if chunk_paths else None
        if temp_dir and os.path.isdir(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass
