"""Step 7 P1 fine-tune on the reviewed A wins (STEP7_PLAN_2026-10-23.md §3; EXPERIMENT_REPORT §22.6, D1).

--mode probe: 20 steps at batch 32 into models/step7_probe_v2; measures step time and peak memory. Its output is discarded
              and never used as P1, so no checkpoint or final model is written (no 12 GB copies); nothing else differs from
              train. probe_v1 (17:18 UTC) trained 20/20 finite steps, then crashed: it skipped only save_model, and the Trainer's
              own save at max_steps wrote trainer_state.json into the checkpoint folder save_model never made.
--mode train: D1 as locked: 2000 steps, batch 32, lr 1e-4, a checkpoint every 500 (all 4 kept) into models/step7_train_v1.
Derived from experiments/bowl_step5/run_training_smoke.py (same audited loads, finite loss/gradient checks, preserved P0
statistics, offline HF, never-overwrite output); the D1 settings and the dataset are the only intended differences.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, choices=('probe', 'train'))
mode = parser.parse_args().mode
ROOT = Path.home() / 'robot-transfer/bowl'
REPO = ROOT / 'Isaac-GR00T'
VERSION = {'probe': 'v2', 'train': 'v1'}[mode]
RUN = ROOT / f'runs/step7_{mode}_{VERSION}'
OUTPUT = ROOT / f'models/step7_{mode}_{VERSION}'
DATASET = ROOT / 'datasets/step7_train_v1'
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
                  WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false')
manifest = json.loads((ROOT / 'models/model-manifest.json').read_text())
snapshots = {m['repo']: m['snapshot'] for m in manifest['models']}
from transformers import AutoModel, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

health_path = RUN / 'finite-training.jsonl'
health_counts = {'loss': 0, 'gradient_norm': 0}
def record_finite(kind, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'Nonfinite {kind}')
    health_counts[kind] += 1
    with health_path.open('a') as f:
        f.write(json.dumps({'kind': kind, 'value': float(value.detach().float().item()), 't': time.time()}) + '\n')

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
        (RUN / 'model-loads.json').write_text(json.dumps(load_report, indent=2, default=str))
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
if mode == 'probe':
    Gr00tTrainer.save_model = lambda self, *args, **kwargs: print('PROBE_SKIPS_FINAL_SAVE', flush=True)
    original_maybe_save = Gr00tTrainer._maybe_log_save_evaluate
    def no_checkpoint(self, *args, **kwargs):
        if self.control.should_save:
            print('PROBE_SKIPS_CHECKPOINT', self.state.global_step, flush=True)
            self.control.should_save = False
        return original_maybe_save(self, *args, **kwargs)
    Gr00tTrainer._maybe_log_save_evaluate = no_checkpoint
original_run = experiment.run
def run_with_preserved_statistics(config):
    config.data.override_pretraining_statistics = False
    config.training.logging_steps = 1 if mode == 'probe' else 10
    (RUN / 'resolved-config.json').write_text(json.dumps({'data': vars(config.data), 'model': vars(config.model),
                                                          'training': vars(config.training)}, indent=1, default=str))
    print('P1_OVERRIDES preserve_P0_statistics=True logging_steps', config.training.logging_steps, flush=True)
    return original_run(config)
experiment.run = run_with_preserved_statistics

assert not OUTPUT.exists(), 'Never resume/overwrite implicitly'
steps, save = (20, 1000) if mode == 'probe' else (2000, 500)
sys.argv = ['launch_finetune.py',
    '--base-model-path', snapshots['nvidia/GR00T-N1.7-DROID'],
    '--dataset-path', str(DATASET),
    '--embodiment-tag', 'OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT',
    '--modality-config-path', str(ROOT / 'experiments/bowl_step5/checkpoint_modalities.py'),
    '--output-dir', str(OUTPUT), '--num-gpus', '1', '--global-batch-size', '32',
    '--max-steps', str(steps), '--save-steps', str(save), '--save-total-limit', '4',
    '--learning-rate', '0.0001', '--warmup-ratio', '0.05', '--weight-decay', '0.00001',
    '--state-dropout-prob', '0.2', '--shortest-image-edge', '256', '--crop-fraction', '0.95',
    '--save-only-model']
# Left at the upstream FinetuneConfig defaults on purpose: color_jitter_params None (P0's own jitter), use_percentiles True,
# tune_llm/tune_visual False, tune_projector/tune_diffusion_model True, dataloader_num_workers 2, shard_size 1024,
# episode_sampling_rate 0.1 (every frame is used; it only sets the interleave), num_shards_per_epoch 1e5.
print('P1_TRAIN_ARGS', json.dumps(sys.argv), flush=True)
t0 = time.time()
runpy.run_path(str(REPO / 'gr00t/experiment/launch_finetune.py'), run_name='__main__')
wall = time.time() - t0
assert health_counts['loss'] >= steps and health_counts['gradient_norm'] >= steps, health_counts
summary = {'mode': mode, 'steps': steps, 'wall_s': round(wall, 1), 'finite_checks': health_counts,
           'peak_memory_allocated_GiB': round(torch.cuda.max_memory_allocated() / 2**30, 2),
           'peak_memory_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2**30, 2),
           'checkpoints': sorted(p.name for p in OUTPUT.glob('checkpoint-*'))}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
print('FINITE_TRAINING_CHECKS', json.dumps(summary), flush=True)
