"""Read-only proof that the disposable two-update checkpoint is usable evidence."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT = Path.home() / "robot-transfer/bowl"
sys.path.insert(0, str(ROOT / "Isaac-GR00T"))
from gr00t.data.types import ModalityConfig
RUN = ROOT / "runs/step5_training_smoke_v1"
manifest = json.loads((ROOT / "models/model-manifest.json").read_text())
p0_entry = next(x for x in manifest["models"] if x["repo"] == "nvidia/GR00T-N1.7-DROID")
p0 = Path(p0_entry["snapshot"])
candidate = ROOT / "models/disposable_smoke_v1/checkpoint-2"
report = {"candidate": str(candidate), "purpose": "disposable pipeline verification, not P1"}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

assert (RUN / "exit-code.txt").read_text().strip() == "0"
state = json.loads((candidate / "trainer_state.json").read_text())
assert state["global_step"] == 2
report["global_step"] = state["global_step"]
health = [json.loads(line) for line in (RUN / "finite-training.jsonl").read_text().splitlines()]
assert [x["kind"] for x in health] == ["loss", "gradient_norm"] * 2
assert all(torch.isfinite(torch.tensor(x["value"])) for x in health)
report["finite_training"] = health

report["p0_files_unchanged"] = []
for item in p0_entry["files"]:
    value = sha256(p0 / item["path"])
    assert value == item["sha256"], item["path"]
    report["p0_files_unchanged"].append({"path": item["path"], "sha256": value})

original_stats = json.loads((p0 / "statistics.json").read_text())
candidate_stats = json.loads((candidate / "statistics.json").read_text())
assert original_stats == candidate_stats, "Normalization changed"
report["processor_statistics_equal"] = True
original_proc = json.loads((p0 / "processor_config.json").read_text())
candidate_proc = json.loads((candidate / "processor_config.json").read_text())
def canonical_modalities(processor):
    return {tag: {name: ModalityConfig(**cfg) for name, cfg in configs.items()}
            for tag, configs in processor["processor_kwargs"]["modality_configs"].items()}
assert canonical_modalities(original_proc) == canonical_modalities(candidate_proc)
report["modality_configs_equal_after_default_expansion"] = True
report["processor_serialization_note"] = "Candidate writes explicit default null modality fields; smoke intentionally set training state_dropout_prob=0 instead of 0.2."

old_map = json.loads((p0 / "model.safetensors.index.json").read_text())["weight_map"]
new_map = json.loads((candidate / "model.safetensors.index.json").read_text())["weight_map"]
missing = set(old_map) - set(new_map)
assert missing == {"backbone.model.lm_head.weight"} and not (set(new_map) - set(old_map))
embedding_key = "backbone.model.model.language_model.embed_tokens.weight"
head_key = "backbone.model.lm_head.weight"
with safe_open(p0 / old_map[head_key], framework="pt", device="cpu") as f:
    head = f.get_tensor(head_key)
with safe_open(p0 / old_map[embedding_key], framework="pt", device="cpu") as f:
    embedding = f.get_tensor(embedding_key)
assert torch.equal(head, embedding), "Omitted LM head is not identical to input embedding"
cosmos = next(x for x in manifest["models"] if x["repo"] == "nvidia/Cosmos-Reason2-2B")
cosmos_config = json.loads((Path(cosmos["snapshot"]) / "config.json").read_text())
assert cosmos_config["tie_word_embeddings"] and cosmos_config["text_config"]["tie_word_embeddings"]
report["omitted_tied_alias"] = {"omitted": head_key, "retained": embedding_key,
    "p0_tensors_exactly_equal": True, "pinned_backbone_ties_embeddings": True,
    "runtime_pointer_verification": "required in fresh load report"}
del head, embedding
report["state_dict_key_counts"] = {"p0": len(old_map), "candidate": len(new_map)}
report["weight_comparisons"] = []
for key in ["action_head.action_decoder.layer1.W", "action_head.action_decoder.layer2.W"]:
    with safe_open(p0 / old_map[key], framework="pt", device="cpu") as f:
        old = f.get_tensor(key).float()
    with safe_open(candidate / new_map[key], framework="pt", device="cpu") as f:
        new = f.get_tensor(key).float()
    assert old.shape == new.shape and torch.isfinite(new).all()
    diff = (new - old).abs()
    changed = int(torch.count_nonzero(diff))
    assert changed > 0, key
    report["weight_comparisons"].append({"key": key, "shape": list(new.shape),
        "changed_elements": changed, "max_abs_difference": float(diff.max())})

report["candidate_files"] = [{"path": p.name, "size": p.stat().st_size, "sha256": sha256(p)}
                             for p in sorted(candidate.iterdir()) if p.is_file()]
(RUN / "checkpoint-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("candidate_files", "p0_files_unchanged")}, indent=2))
