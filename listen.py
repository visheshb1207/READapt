
# LISTEN MODULE (Speech Analysis)
# File: listen.py


import os
import uuid
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from jiwer import process_words
from phonemizer import phonemize
import whisper
from typing import Dict, List, Any, Optional
import time


# CONFIG


SAMPLE_RATE = 16000
AUDIO_FOLDER = "data/audio/"
WHISPER_MODEL_SIZE = "base"

os.makedirs(AUDIO_FOLDER, exist_ok=True)


# SAFE MODEL LOADING


try:
    _asr_model = whisper.load_model(WHISPER_MODEL_SIZE)
except Exception as e:
    print(" Whisper model failed to load:", e)
    _asr_model = None



# DYSLEXIC PHONEME GROUPS


DYSLEXIA_PHONEME_GROUPS = {
    ("b", "d"): "mirror_confusion",
    ("p", "b"): "voicing_confusion",
    ("f", "v"): "voicing_confusion",
    ("t", "θ"): "fricative_error",
    ("m", "n"): "nasal_confusion",
    ("s", "z"): "voicing_confusion"
}



# AUDIO RECORDING (FIXED DURATION)


def record_audio(duration: int = 6) -> str:
    """
    Records microphone safely (Streamlit compatible)
    """
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()

    filename = f"{AUDIO_FOLDER}{uuid.uuid4().hex[:8]}.wav"
    write(filename, SAMPLE_RATE, audio)

    return filename



# CONTINUOUS RECORDING MODE


def start_continuous_recording():
    """Start live microphone stream"""
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    stream.start()
    return stream, []


def process_continuous_audio(stream, buffer_list, chunk_duration=0.5):
    """Collect chunks during reading"""
    frames = int(chunk_duration * SAMPLE_RATE)
    data, _ = stream.read(frames)
    buffer_list.append(data.copy())


def stop_continuous_recording(stream, buffer_list) -> str:
    """Stop recording and save audio"""
    stream.stop()
    stream.close()

    audio = np.concatenate(buffer_list, axis=0)

    filename = f"{AUDIO_FOLDER}{uuid.uuid4().hex[:8]}.wav"
    write(filename, SAMPLE_RATE, audio)

    return filename



# TRANSCRIPTION


def transcribe_audio(audio_path: str) -> str:
    """Speech → Text using Whisper"""
    if _asr_model is None:
        return ""

    try:
        result = _asr_model.transcribe(audio_path)
        return result.get("text", "").strip().lower()
    except Exception as e:
        print("Transcription failed:", e)
        return ""



# ALIGNMENT


def align_spoken_to_target(target_text: str, spoken_text: str) -> Dict:
    result = process_words(target_text.lower(), spoken_text.lower())

    return {
        "wer": round(result.wer, 3),
        "substitutions": int(result.substitutions),
        "insertions": int(result.insertions),
        "deletions": int(result.deletions)
    }



# PHONETIC ERROR DETECTION (FIXED)


def detect_phonetic_errors(target_word: str, spoken_word: str) -> List[Dict]:
    try:
        target_ph = phonemize(target_word, language="en-us", backend="espeak").strip().split()
        spoken_ph = phonemize(spoken_word, language="en-us", backend="espeak").strip().split()
    except Exception:
        return []

    errors = []

    for t, s in zip(target_ph, spoken_ph):
        for pair, err_type in DYSLEXIA_PHONEME_GROUPS.items():
            if t in pair and s in pair and t != s:
                errors.append({
                    "target_phoneme": t,
                    "spoken_phoneme": s,
                    "type": err_type
                })

    return errors



# CONFIDENCE FUNCTIONS


def compute_word_confidence(base_confidence: float, phonetic_error_count: int) -> float:
    penalty = min(0.12 * phonetic_error_count, 0.4)
    return round(max(base_confidence - penalty, 0.0), 2)


def compute_sentence_confidence(avg_word_confidence: float, wer: float) -> float:
    penalty = min(wer, 0.5)
    return round(avg_word_confidence * (1 - penalty), 2)


def classify_error_type(phonetic_errors: List[Dict], focus_drop: bool = False) -> str:
    if len(phonetic_errors) >= 2 and focus_drop:
        return "dyslexic_error"
    elif len(phonetic_errors) == 1:
        return "possible_accent"
    elif len(phonetic_errors) > 2:
        return "pronunciation_error"
    return "normal"



# MAIN LISTEN PIPELINE


def run_listen_module(
    target_text: str,
    focus_drop: bool = False,
    record_duration: int = 6
) -> Dict[str, Any]:

    # Record speech
    audio_path = record_audio(record_duration)

    # Transcribe
    spoken_text = transcribe_audio(audio_path)

    # Align
    alignment = align_spoken_to_target(target_text, spoken_text)

    # Word Analysis
    target_words = target_text.lower().split()
    spoken_words = spoken_text.lower().split()

    word_results = []
    total_conf = 0

    for i in range(min(len(target_words), len(spoken_words))):
        t = target_words[i]
        s = spoken_words[i]

        phonetic_errors = detect_phonetic_errors(t, s)
        error_type = classify_error_type(phonetic_errors, focus_drop)

        word_conf = compute_word_confidence(
            base_confidence=0.85,
            phonetic_error_count=len(phonetic_errors)
        )

        total_conf += word_conf

        word_results.append({
            "target_word": t,
            "spoken_word": s,
            "confidence": word_conf,
            "phonetic_errors": phonetic_errors,
            "error_type": error_type
        })

    avg_word_conf = total_conf / max(len(word_results), 1)
    sentence_conf = compute_sentence_confidence(avg_word_conf, alignment["wer"])

    return {
        "audio_path": audio_path,
        "spoken_text": spoken_text,
        "expected_text": target_text,
        "wer": alignment["wer"],
        "sentence_confidence": sentence_conf,
        "insertions": alignment["insertions"],
        "deletions": alignment["deletions"],
        "substitutions": alignment["substitutions"],
        "word_results": word_results
    }



# TESTING


if __name__ == "__main__":
    sample_text = "The friendly dragon played gently in the beautiful garden"
    print(" Speak now...")
    result = run_listen_module(sample_text)

    print("\nLISTEN RESULT")
    for k, v in result.items():
        print(f"{k}: {v}")

