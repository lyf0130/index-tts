import json
import os
import subprocess

from faster_whisper import WhisperModel
from opencc import OpenCC


WORK = "work_voicecopy"
OBJECT_VIDEO = os.path.join("tests", "object.mp4")
OBJECT_AUDIO = os.path.join(WORK, "object_audio_16k.wav")
TRANSCRIPT = os.path.join(WORK, "object_transcript.json")


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_object_audio():
    os.makedirs(WORK, exist_ok=True)
    run([
        "ffmpeg",
        "-y",
        "-i",
        OBJECT_VIDEO,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        OBJECT_AUDIO,
    ])


def main():
    extract_object_audio()

    cc = OpenCC("t2s")
    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
    segments, info = model.transcribe(OBJECT_AUDIO, beam_size=5, vad_filter=True)

    items = []
    for segment in segments:
        text = cc.convert(segment.text.strip())
        if not text:
            continue
        items.append({
            "start": round(float(segment.start), 2),
            "end": round(float(segment.end), 2),
            "text": text,
        })

    data = {
        "language": info.language,
        "probability": info.language_probability,
        "segments": items,
    }
    with open(TRANSCRIPT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} simplified Chinese segments to {TRANSCRIPT}")


if __name__ == "__main__":
    main()
