"""Launch the upstream P0 server unchanged, recording checkpoint load diagnostics."""
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

ROOT = Path.home() / "robot-transfer/bowl"
GROOT = ROOT / "Isaac-GR00T"
sys.path.insert(0, str(GROOT))
os.chdir(GROOT)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="0")
model_manifest = json.loads((ROOT / "models/model-manifest.json").read_text())
snapshots = {m["repo"]: m for m in model_manifest["models"]}
cosmos = snapshots["nvidia/Cosmos-Reason2-2B"]
ref = Path(cosmos["snapshot"]).parent.parent / "refs/main"
ref.parent.mkdir(exist_ok=True)
ref.write_text(cosmos["revision"])

import numpy as np
import torch
from transformers import AutoModel, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.utils.hub import cached_file

resolved_config = cached_file("nvidia/Cosmos-Reason2-2B", "config.json", local_files_only=True)
assert cosmos["revision"] in resolved_config, resolved_config
policy_seed = int(os.environ.get("POLICY_SEED", "0"))
effective_checkpoint = os.environ.get("POLICY_CHECKPOINT", snapshots["nvidia/GR00T-N1.7-DROID"]["snapshot"])
policy_label = "disposable copy" if "POLICY_CHECKPOINT" in os.environ else "P0"
random.seed(policy_seed)
np.random.seed(policy_seed)
torch.manual_seed(policy_seed)
report = {"utc": datetime.now(timezone.utc).isoformat(), "offline": True,
          "cosmos_config": resolved_config, "cosmos_revision": cosmos["revision"],
          "p0_snapshot": snapshots["nvidia/GR00T-N1.7-DROID"]["snapshot"],
          "effective_checkpoint": effective_checkpoint, "policy_label": policy_label,
          "seed": policy_seed, "loads": []}

def instrument(cls, label):
    original = cls.from_pretrained
    def audited(*args, **kwargs):
        # Diagnostics only: return the same model to the unmodified upstream caller.
        model, info = original(*args, **kwargs, output_loading_info=True)
        entry = {"label": label, "path": str(args[0]), "loading_info": info,
                 "config_commit": getattr(model.config, "_commit_hash", None)}
        if label == "disposable copy":
            backbone = model.backbone.model
            entry["lm_head_and_input_embedding_share_storage"] = (
                backbone.get_input_embeddings().weight.data_ptr() == backbone.get_output_embeddings().weight.data_ptr())
            assert entry["lm_head_and_input_embedding_share_storage"], "Expected tied output-head alias missing"
        report["loads"].append(entry)
        (ROOT / "logs/model-loading.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print("CHECKPOINT_LOAD_DIAGNOSTICS", json.dumps(entry, default=str), flush=True)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if info.get(key):
                raise RuntimeError(f"Review required: {label} {key}: {info[key]}")
        return model
    cls.from_pretrained = staticmethod(audited)

instrument(AutoModel, policy_label)
instrument(Qwen3VLForConditionalGeneration, "Cosmos")
# Transformers 4.57.3 performs a model-info query for remote-named tokenizers
# even in offline mode. Resolve this processor to the identical pinned local
# snapshot, preserving every tokenizer and model setting.
original_processor_load = Qwen3VLProcessor.from_pretrained
def local_processor_load(path, *args, **kwargs):
    if str(path) == "nvidia/Cosmos-Reason2-2B":
        path = cosmos["snapshot"]
    print("PINNED_PROCESSOR_PATH", str(path), flush=True)
    return original_processor_load(path, *args, **kwargs)
Qwen3VLProcessor.from_pretrained = staticmethod(local_processor_load)
from gr00t.eval.run_gr00t_server import ServerConfig, main

print("PINNED_OFFLINE_RUNTIME", json.dumps(report), flush=True)
main(ServerConfig(model_path=effective_checkpoint,
                  embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
                  device="cuda", host="127.0.0.1", port=15555,
                  strict=True, use_sim_policy_wrapper=True))
