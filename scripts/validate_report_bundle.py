"""Validate a portable report independently of its original workspace files."""

import argparse
import hashlib
import json
import stat
import tempfile
import zipfile
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(
            value for key, value in attrs if key in {"href", "src", "poster"} and value
        )


def validate(archive_path, decode_media=False):
    archive_path = Path(archive_path).resolve()
    media = []
    with tempfile.TemporaryDirectory(prefix="so101-report-check-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert len(names) == len(set(names)), "Duplicate archive paths"
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                assert not path.is_absolute() and ".." not in path.parts, member.filename
                assert not stat.S_ISLNK(member.external_attr >> 16), member.filename
                assert path.suffix not in {".pt", ".npz"}, member.filename
                assert not any(part == ".env" or part.startswith(".env.") for part in path.parts)
            assert archive.testzip() is None, "Archive CRC failure"
            manifest = json.loads(archive.read("bundle_manifest.json"))
            for record in manifest["files"]:
                data = archive.read(record["path"])
                assert len(data) == record["bytes"], record["path"]
                assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
            archive.extractall(extracted)
        checked = 0
        missing = []
        for document in extracted.rglob("*.html"):
            parser = Links()
            parser.feed(document.read_text())
            for target in parser.targets:
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                path = (document.parent / unquote(parsed.path)).resolve()
                checked += 1
                if not path.is_relative_to(extracted) or not path.exists():
                    missing.append(
                        {"document": str(document.relative_to(extracted)), "target": target}
                    )
        assert not missing, missing
        videos = sorted(extracted.rglob("*.mp4"))
        if decode_media:
            import imageio.v2 as imageio

            for path in videos:
                reader = imageio.get_reader(path)
                try:
                    count = reader.count_frames()
                    metadata = reader.get_meta_data()
                    assert count > 0, path
                    for index in (0, count // 2, count - 1):
                        frame = reader.get_data(index)
                        assert frame.ndim == 3 and frame.shape[-1] == 3, path
                    media.append(
                        {
                            "path": str(path.relative_to(extracted)),
                            "frames": count,
                            "fps": metadata["fps"],
                            "size": list(metadata["size"]),
                            "decoded_first_middle_last": True,
                        }
                    )
                finally:
                    reader.close()
        with archive_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "checked_at": datetime.now(UTC).isoformat(),
            "bundle": str(archive_path),
            "bundle_sha256": digest,
            "bundle_bytes": archive_path.stat().st_size,
            "bundle_files": len(names),
            "entrypoint": manifest["entrypoint"],
            "report_path": manifest["report"],
            "local_links_checked_after_extraction": checked,
            "missing_links": missing,
            "mp4_files_in_archive": len(videos),
            "media_decoded": len(media),
            "archive_crc_verified": True,
            "bundle_manifest_file_hashes_verified": len(manifest["files"]),
            "archive_paths_safe": True,
            "excluded_checkpoints_npz_dotenv_files": True,
            "validation_scope": "Independent extraction, HTML links, CRC and per-file SHA256; optional first/middle/last video decoding. No simulator rollout.",
            "videos": media,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", type=Path, nargs="?", default=Path("artifacts/plug_analysis/plug-report.zip")
    )
    parser.add_argument("--decode-media", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/plug_analysis/portable_validation.json")
    )
    args = parser.parse_args()
    result = validate(args.archive, args.decode_media)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Validated {result['bundle_files']} files, {result['local_links_checked_after_extraction']} links, {result['media_decoded']} decoded videos; SHA256 {result['bundle_sha256']}"
    )


if __name__ == "__main__":
    main()
