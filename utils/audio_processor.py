import yt_dlp
from pydub import AudioSegment
import os
from pathlib import Path
from uuid import uuid4

DOWNLOAD_DIR = Path(__file__).resolve().parents[1] / "downloades"
DOWNLOAD_DIR.mkdir(exist_ok=True)

def download_youtube_audio(url :str) ->str:
    run_dir = DOWNLOAD_DIR / uuid4().hex
    run_dir.mkdir()
    output_path = str(run_dir / "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
    wav_files = list(run_dir.glob("*.wav"))
    if not wav_files:
        raise RuntimeError("YouTube audio download finished but FFmpeg did not create a WAV file. Install FFmpeg and retry.")
    return str(wav_files[0])



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks


def cleanup_audio_files(chunks: list[str]) -> None:
    """Delete only generated audio for a completed/failed run."""
    if not chunks:
        return
    for chunk in chunks:
        try:
            Path(chunk).unlink(missing_ok=True)
        except OSError as error:
            print(f"Could not remove temporary chunk {chunk}: {error}")

    # YouTube downloads are stored in their own UUID directory, so this is safe.
    parent = Path(chunks[0]).parent
    try:
        if parent.parent == DOWNLOAD_DIR:
            for file in parent.iterdir():
                file.unlink(missing_ok=True)
            parent.rmdir()
    except OSError as error:
        print(f"Could not remove temporary download directory: {error}")

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Local file does not exist: {source}")
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
