"""Download pinned snapshots using existing HF credential store; never print credentials."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path.home() / "robot-transfer/bowl"
models = [
    ("nvidia/GR00T-N1.7-DROID", "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5"),
    ("nvidia/Cosmos-Reason2-2B", "9ce19a195e423419c349abfc86fd07178b230561"),
]
manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "models": []}
for repo, revision in models:
    print(f"Downloading pinned {repo} at {revision}", flush=True)
    location = Path(snapshot_download(repo_id=repo, revision=revision, max_workers=4))
    entries = []
    for file in sorted(location.rglob("*")):
        if not file.is_file():
            continue
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        entries.append({"path": str(file.relative_to(location)), "size": file.stat().st_size, "sha256": digest.hexdigest()})
    manifest["models"].append({"repo": repo, "revision": revision, "snapshot": str(location), "files": entries})
    (root / "models/model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {repo}: {len(entries)} files", flush=True)
manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
(root / "models/model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("PINNED_MODELS_COMPLETE", flush=True)
