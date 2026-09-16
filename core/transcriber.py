import whisper
import os
import requests
from pydub import AudioSegment
from core.runtime import invoke_with_retry

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():

    global _model  

    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL) 
        print("Whisper model loaded.")
    return _model 


def transcribe_chunk_whisper(chunk_path: str, language: str | None = None) -> str:
    if not os.path.isfile(chunk_path):
        raise FileNotFoundError(f"Audio chunk was not found: {chunk_path}")
    model = load_model()
    # Let Whisper auto-detect Hinglish; forcing Hindi can hurt English portions.
    result = model.transcribe(chunk_path, task="transcribe", language=language, fp16=False)
    return result["text"]  


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        def request_transcription():
            f.seek(0)
            response = requests.post(
                SARVAM_STT_TRANSLATE_URL, headers=headers, files=files,
                data=data, timeout=(15, 120),
            )
            # Rate limiting and server errors are retryable.
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            return response

        response = invoke_with_retry(request_transcription, "Sarvam transcription")

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()

   



def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam when configured, otherwise local Whisper auto-detection
    """
    if language.lower() == "hinglish" and SARVAM_API_KEY:
        return transcribe_chunk_sarvam(chunk_path)
    if language.lower() == "hinglish":
        print("SARVAM_API_KEY is not set; using local Whisper auto-detection for Hinglish.")
        return transcribe_chunk_whisper(chunk_path, language=None)
    return transcribe_chunk_whisper(chunk_path, language="en")


def transcribe_all(chunks: list, language: str = "english") -> str:

    full_transcript = "" 

    engine = "Sarvam AI" if language.lower() == "hinglish" and SARVAM_API_KEY else "Whisper (Hinglish auto-detect)" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):  

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language=language)

        full_transcript += text + " "  

    print("Transcription complete.")

    return full_transcript.strip()  
