import json
import os
import subprocess

VIDEO_DURATION = 59.65
WORK = "work_voicecopy"
SEG_DIR = os.path.join(WORK, "tts_segments")
FIT_DIR = os.path.join(WORK, "fit_segments")
TRANSCRIPT = os.path.join(WORK, "object_transcript.json")
os.makedirs(FIT_DIR, exist_ok=True)


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def probe_duration(path):
    out = subprocess.check_output([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ], text=True)
    return float(json.loads(out)["format"]["duration"])


def make_silence(path, duration):
    run([
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r=48000:cl=mono",
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "pcm_s16le",
        path,
    ])


def atempo_filter(ratio):
    # Keep every atempo value inside a conservative ffmpeg range.
    parts = []
    while ratio > 2.0:
        parts.append("atempo=2.0")
        ratio /= 2.0
    while ratio < 0.5:
        parts.append("atempo=0.5")
        ratio /= 0.5
    parts.append(f"atempo={ratio:.6f}")
    return ",".join(parts)


def load_segments():
    with open(TRANSCRIPT, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    segments = []
    previous_end = 0.0
    for idx, item in enumerate(transcript.get("segments", [])):
        start = float(item["start"])
        end = float(item["end"])
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if end <= start:
            raise ValueError(f"Segment {idx} end must be greater than start: {start} -> {end}")
        if start < previous_end:
            raise ValueError(f"Segment {idx} overlaps previous segment: {start} < {previous_end}")
        segments.append((start, end, text))
        previous_end = end

    if not segments:
        raise ValueError(f"No usable segments found in {TRANSCRIPT}")
    return segments


def ducking_expression(segments):
    return "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end, _ in segments)


def main():
    segments = load_segments()
    concat_paths = []
    cursor = 0.0

    for idx, (start, end, _) in enumerate(segments):
        gap = max(0.0, start - cursor)
        if gap > 0.005:
            silence = os.path.join(FIT_DIR, f"gap_{idx:02d}.wav")
            make_silence(silence, gap)
            concat_paths.append(silence)

        src = os.path.join(SEG_DIR, f"seg_{idx:02d}.wav")
        target_duration = end - start
        src_duration = probe_duration(src)
        ratio = src_duration / target_duration
        fit = os.path.join(FIT_DIR, f"seg_{idx:02d}_fit.wav")
        filt = f"{atempo_filter(ratio)},apad,atrim=0:{target_duration:.3f}"
        run([
            "ffmpeg",
            "-y",
            "-i",
            src,
            "-af",
            filt,
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            fit,
        ])
        concat_paths.append(fit)
        cursor = end

    tail = max(0.0, VIDEO_DURATION - cursor)
    if tail > 0.005:
        silence = os.path.join(FIT_DIR, "tail.wav")
        make_silence(silence, tail)
        concat_paths.append(silence)

    concat_file = os.path.join(WORK, "concat_audio.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for path in concat_paths:
            abs_path = os.path.abspath(path).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    voice_track = os.path.join(WORK, "object_voicecopied_track.wav")
    run([
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-ar",
        "48000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        voice_track,
    ])

    final_video = os.path.join("tests", "object_voicecopied.mp4")
    run([
        "ffmpeg",
        "-y",
        "-i",
        os.path.join("tests", "object.mp4"),
        "-i",
        voice_track,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        final_video,
    ])
    print(final_video)

    with_bg_video = os.path.join("tests", "object_voicecopied_with_bg.mp4")
    filter_complex = (
        f"[0:a]volume=enable='{ducking_expression(segments)}':volume=0.08[bg];"
        "[1:a]volume=1.4[vc];"
        "[bg][vc]amix=inputs=2:duration=first:dropout_transition=0,alimiter=limit=0.95[a]"
    )
    run([
        "ffmpeg",
        "-y",
        "-i",
        os.path.join("tests", "object.mp4"),
        "-i",
        voice_track,
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        with_bg_video,
    ])
    print(with_bg_video)


if __name__ == "__main__":
    main()
