"""Independently validate the latest report bundle, local links and decoded videos."""

import base64
import hashlib
import json
import tempfile
import zipfile
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

import imageio.v2 as imageio


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []
        self.videos = 0
        self.embedded = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        self.links += [attrs[k] for k in ["href", "src", "poster"] if k in attrs]
        self.videos += tag == "video"
        for key in ["src", "data-policy"]:
            if attrs.get(key, "").startswith("data:"):
                self.embedded.append(attrs[key])


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts/training_results"
    videos = []
    with tempfile.TemporaryDirectory(prefix="latest-report-") as folder:
        extracted = Path(folder)
        with zipfile.ZipFile(out / "training-results.zip") as archive:
            assert archive.testzip() is None
            for name in archive.namelist():
                assert (extracted / name).resolve().is_relative_to(extracted)
            archive.extractall(extracted)
        manifest = json.loads((extracted / "manifest.json").read_text())
        for item in manifest:
            data = (extracted / item["path"]).read_bytes()
            assert len(data) == item["bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
        report = extracted / "artifacts/training_results/index.html"
        parser = Parser()
        parser.feed(report.read_text())
        assert len(parser.ids) == len(set(parser.ids))
        assert {"plug", "push", "videos", "sources"} <= set(parser.ids)
        assert "background:#fff" in report.read_text()
        analysis = json.loads((report.parent / "latest_analysis.json").read_text())
        assert parser.videos == len(analysis["media"])
        assert len(analysis["runs"]) == 18
        assert analysis["checks"]["final_test_episodes"] == 36864
        embedded_video_hashes = set()
        embedded_images = 0
        for value in parser.embedded:
            metadata, encoded = value.split(",", 1)
            data = base64.b64decode(encoded, validate=True)
            if metadata == "data:video/webm;base64":
                embedded_video_hashes.add(hashlib.sha256(data).hexdigest())
            elif metadata == "data:image/svg+xml;base64":
                ElementTree.fromstring(data)
                embedded_images += 1
        assert embedded_images == 5 + len(analysis.get("repair", {}).get("plots", []))
        assert len(embedded_video_hashes) == 2 * len(analysis["media"])
        media = set()
        links = 0
        for value in parser.links:
            parts = urlsplit(value)
            if parts.scheme:
                continue
            if not parts.path:
                assert parts.fragment in parser.ids
                continue
            if parts.path == "training-results.zip":
                continue  # Download points to the archive itself in the workspace.
            path = (report.parent / unquote(parts.path)).resolve()
            assert path.is_relative_to(extracted) and path.is_file(), value
            links += 1
            if path.suffix == ".svg":
                ElementTree.parse(path)
            if path.suffix == ".mp4":
                media.add(path)
        media = set((report.parent / "media").glob("*.webm"))
        assert len(media) == 2 * len(analysis["media"])
        for path in media:
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() in embedded_video_hashes
        for i, path in enumerate(sorted(media)):
            reader = imageio.get_reader(path)
            try:
                count = reader.count_frames()
                metadata = reader.get_meta_data()
                assert count > 0 and abs(metadata["fps"] - 25) < 1e-6
                for index in [0, count // 2, count - 1]:
                    frame = reader.get_data(index)
                    assert frame.ndim == 3 and frame.shape[-1] == 3
                videos.append(
                    dict(
                        path=str(path.relative_to(extracted)),
                        frames=count,
                        fps=metadata["fps"],
                        size=list(metadata["size"]),
                        decoded=True,
                    )
                )
            finally:
                reader.close()
            if (i + 1) % 24 == 0:
                print(f"Decoded {i + 1}/{len(media)} videos", flush=True)
    result = dict(
        validated_at=datetime.now(UTC).isoformat(),
        files_hash_verified=len(manifest),
        local_links_checked=links,
        video_examples=parser.videos,
        embedded_videos_verified=len(embedded_video_hashes),
        embedded_plots_verified=embedded_images,
        videos_decoded=len(videos),
        archive_crc_passed=True,
        videos=videos,
    )
    (out / "portable_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "validation.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "videos"}, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "videos"}, indent=2))


if __name__ == "__main__":
    main()
