"""Cache browser-portable VP9 copies; preserve original full-resolution MP4 evidence."""

import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image

root = Path(__file__).resolve().parents[1]
out = root / "artifacts/training_results"
folder = out / "media"
folder.mkdir(exist_ok=True)
exe = imageio_ffmpeg.get_ffmpeg_exe()


def convert(relative):
    source = (out / relative).resolve()
    name = hashlib.sha256(relative.encode()).hexdigest()[:16] + ".webm"
    dest = folder / name
    if not dest.exists():
        temp = dest.with_suffix(".tmp.webm")
        cmd = [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            "scale=min(960\\,iw):-2",
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "34",
            "-b:v",
            "0",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-row-mt",
            "1",
            "-threads",
            "2",
            str(temp),
        ]
        subprocess.run(cmd, check=True)
        temp.replace(dest)
    poster = dest.with_suffix(".jpg")
    if not poster.exists():
        reader = imageio.get_reader(dest)
        try:
            frame = Image.fromarray(reader.get_data(0))
            frame.thumbnail((640, 360))
            frame.save(poster, quality=80)
        finally:
            reader.close()
    return dict(
        source=relative,
        webm=str(dest.relative_to(out)),
        bytes=dest.stat().st_size,
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    )


def main():
    analysis = json.loads((out / "latest_analysis.json").read_text())
    paths = sorted({m[key] for m in analysis["media"] for key in ["outside", "policy"]})
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for r in pool.map(convert, paths):
            records.append(r)
            if len(records) % 12 == 0:
                print(f"Encoded {len(records)}/{len(paths)}", flush=True)
    (folder / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print("DONE", sum(r["bytes"] for r in records), flush=True)


if __name__ == "__main__":
    main()
