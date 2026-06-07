from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ORIGINAL_URL = "http://people.cs.nctu.edu.tw/~yllee/yllee&whtsai_sfv.html"
KODAK = "https://r0k.us/graphics/kodak/kodak/kodim{number:02d}.png"


def collect(output: str, count: int = 12) -> dict:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "original_paper_dataset_url": ORIGINAL_URL,
        "original_status": "unavailable; direct server returned no content and archive capture was not found",
        "fallback": "Kodak Lossless True Color Image Suite",
        "fallback_source": "https://r0k.us/graphics/kodak/",
        "files": [],
    }
    for number in range(1, count + 1):
        url = KODAK.format(number=number)
        path = root / f"kodim{number:02d}.png"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        manifest["files"].append(
            {
                "name": path.name,
                "url": url,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest

