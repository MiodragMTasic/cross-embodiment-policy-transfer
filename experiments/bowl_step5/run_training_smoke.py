"""Two-step disposable continuation test using the upstream training pipeline."""
import json
import os
from pathlib import Path
import runpy
import sys
import torch

ROOT = Path.home() / 'robot-transfer/bowl'
REPO = ROOT / 'Isaac-GR00T'
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
                  WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false')
manifest = json.loads((ROOT / 'models/model-manifest.json').read_text())
snapshots = {m['repo']: m['snapshot'] for m in manifest['models']}
from transformers import AutoModel, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

health_path = ROOT / 'runs/step5_training_smoke_v1/finite-training.jsonl'
health_counts = {'loss': 0, 'gradient_norm': 0}
def record_finite(kind, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'Nonfinite {kind}')
    health_counts[kind] += 1
    with health_path.open('a') as f:
        f.write(json.dumps({'kind': kind, 'value': float(value.detach().float().item())}) + '\n')

original_clip = torch.nn.utils.clip_grad_norm_
def checked_clip(*args, **kwargs):
    kwargs['error_if_nonfinite'] = True
    value = original_clip(*args, **kwargs)
    record_finite('gradient_norm', value)
    return value
torch.nn.utils.clip_grad_norm_ = checked_clip

load_report = []
def audit(cls, label):
    original = cls.from_pretrained
    def load(*args, **kwargs):
        want_info = kwargs.pop('output_loading_info', False)
        model, info = original(*args, **kwargs, output_loading_info=True)
        load_report.append({'label': label, 'path': str(args[0]), 'info': info})
        (ROOT / 'runs/step5_training_smoke_v1/model-loads.json').write_text(json.dumps(load_report, indent=2, default=str))
        for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
            if info.get(key):
                raise RuntimeError(f'{label} {key}: {info[key]}')
        return (model, info) if want_info else model
    cls.from_pretrained = staticmethod(load)
audit(AutoModel, 'P0')
audit(Qwen3VLForConditionalGeneration, 'Cosmos')
original_processor = Qwen3VLProcessor.from_pretrained
def processor(path, *args, **kwargs):
    if str(path) == 'nvidia/Cosmos-Reason2-2B':
        path = snapshots['nvidia/Cosmos-Reason2-2B']
    return original_processor(path, *args, **kwargs)
Qwen3VLProcessor.from_pretrained = staticmethod(processor)

from gr00t.experiment import experiment
from gr00t.experiment.trainer import Gr00tTrainer
original_compute_loss = Gr00tTrainer.compute_loss
def checked_loss(self, *args, **kwargs):
    result = original_compute_loss(self, *args, **kwargs)
    value = result[0] if isinstance(result, tuple) else result
    record_finite('loss', value)
    return result
Gr00tTrainer.compute_loss = checked_loss
original_run = experiment.run
def run_with_preserved_statistics(config):
    config.data.override_pretraining_statistics = False
    config.training.logging_steps = 1
    print('SMOKE_OVERRIDES preserve_P0_statistics=True logging_steps=1', flush=True)
    return original_run(config)
experiment.run = run_with_preserved_statistics

output = ROOT / 'models/disposable_smoke_v1'
assert not output.exists(), 'Never resume/overwrite implicitly'
sys.argv = ['launch_finetune.py',
    '--base-model-path', snapshots['nvidia/GR00T-N1.7-DROID'],
    '--dataset-path', str(ROOT / 'datasets/disposable_smoke_v1'),
    '--embodiment-tag', 'OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT',
    '--modality-config-path', str(Path(__file__).with_name('checkpoint_modalities.py')),
    '--output-dir', str(output), '--num-gpus', '1', '--global-batch-size', '1',
    '--dataloader-num-workers', '0', '--max-steps', '2', '--save-steps', '2',
    '--save-total-limit', '1', '--learning-rate', '0.00001', '--warmup-ratio', '0',
    '--state-dropout-prob', '0', '--shortest-image-edge', '256', '--crop-fraction', '0.95',
    '--shard-size', '128', '--num-shards-per-epoch', '1', '--episode-sampling-rate', '1',
    '--save-only-model']
print('DISPOSABLE_SMOKE_ARGS', json.dumps(sys.argv), flush=True)
runpy.run_path(str(REPO / 'gr00t/experiment/launch_finetune.py'), run_name='__main__')
assert health_counts['loss'] >= 2 and health_counts['gradient_norm'] >= 2, health_counts
print('FINITE_TRAINING_CHECKS', json.dumps(health_counts), flush=True)
