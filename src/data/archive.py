"""Create/verify a portable dataset archive with source and protocol hashes."""

import argparse
import json
import tarfile
from pathlib import Path

from src.config import BASE_DIR, DATA_DIR
from src.utils.artifacts import atomic_json, file_sha256


def export_archive(destination: Path) -> None:
    files = (
        sorted((DATA_DIR / "raw").rglob("*.txt"))
        + sorted((DATA_DIR / "v2").rglob("*.csv"))
        + sorted((DATA_DIR / "v2").rglob("*.json"))
    )
    files += [BASE_DIR / "docs" / name for name in ("data_sources.json", "template_overrides.json", "dataset_card.md")]
    manifest = {str(p.relative_to(BASE_DIR)): file_sha256(p) for p in files}
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(BASE_DIR)), recursive=False)

    atomic_json(
        destination.with_suffix(".manifest.json"), {"archive_sha256": file_sha256(destination), "files": manifest}
    )


def verify_archive(path: Path) -> None:
    import hashlib

    manifest = json.loads(path.with_suffix(".manifest.json").read_text())

    if file_sha256(path) != manifest["archive_sha256"]:
        raise ValueError("Archive hash mismatch")

    with tarfile.open(path, "r:gz") as archive:
        if sorted(archive.getnames()) != sorted(manifest["files"]):
            raise ValueError("Archive member list mismatch")

        for member in archive:
            handle = archive.extractfile(member)

            if handle is None or hashlib.sha256(handle.read()).hexdigest() != manifest["files"][member.name]:
                raise ValueError(f"Corrupt archive member: {member.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify_archive(args.path)

    else:
        export_archive(args.path)


if __name__ == "__main__":
    main()
