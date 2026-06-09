import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indextts.infer_v2 import IndexTTS2
from opencc import OpenCC


def load_segments():
    transcript_path = os.path.join("work_voicecopy", "object_transcript.json")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    cc = OpenCC("t2s")
    segments = []
    for idx, item in enumerate(transcript.get("segments", [])):
        start = float(item["start"])
        end = float(item["end"])
        text = cc.convert(str(item.get("text", "")).strip())
        if not text:
            continue
        if end <= start:
            raise ValueError(f"Segment {idx} end must be greater than start: {start} -> {end}")
        segments.append((start, end, text))

    if not segments:
        raise ValueError(f"No usable segments found in {transcript_path}")
    return segments


def main():
    out_dir = os.path.join("work_voicecopy", "tts_segments")
    os.makedirs(out_dir, exist_ok=True)

    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=False,
        device="cpu",
        use_cuda_kernel=False,
        use_deepspeed=False,
    )

    prompt = os.path.join("work_voicecopy", "original_prompt_15s.wav")
    for idx, (_, _, text) in enumerate(load_segments()):
        output = os.path.join(out_dir, f"seg_{idx:02d}.wav")
        print(f">> synth {idx:02d}: {text}")
        tts.infer(
            spk_audio_prompt=prompt,
            text=text,
            output_path=output,
            verbose=True,
            max_text_tokens_per_segment=80,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            num_beams=3,
            repetition_penalty=10.0,
            length_penalty=0.0,
            max_mel_tokens=600,
        )


if __name__ == "__main__":
    main()
