"""Step 7: reviewed A/P0 wins -> one GR00T LeRobot-v2 fine-tuning dataset. Offline, CPU only, never overwrites.

Three subcommands, run on the VM from ~/robot-transfer/bowl with PYTHONDONTWRITEBYTECODE=1 and CUDA_VISIBLE_DEVICES='':
  candidates  any python with numpy (RoboLab/.venv)  locked v5.1 over the explicit case set -> candidate list for the reviewer
  export      RoboLab/.venv (h5py)                   approved wins -> one npz per episode, cut at the v5.1 win control
  build       Isaac-GR00T/.venv (pandas, pyarrow)    npz -> LeRobot v2 dataset + gr00t.data.stats
verify_step7_dataset.py then loads the result through the pinned GR00T loader, processor and collator (CPU).

Inputs are explicit (reviewer N2, STEP7_PLAN D3/D4/S2): case names come only from the collection ledger(s) and the four named
step 5 cases; nothing is globbed. Every name is asserted: never `step8_`, layout 2000-2031, the approved step 5 runner (a3191192).
Only wins listed with verdict OK in the approved-wins file are converted, and the locked v5.1 must reproduce each listed win control.
Each episode keeps rows 0 .. win row: trace row i is control i+1, so a win at control S keeps S rows (as case 2002: 301 of 301).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path.home() / 'robot-transfer/bowl'
HERE = Path(__file__).resolve().parent
TASK = 'BananaInBowlUniformInitPose10cmTask'
TRAINING_LAYOUTS = range(2000, 2032)
RUNNER = ('experiments/bowl_step5/run_controlled_episode_v4_registered.py',
          'a319119268b3eaf93934c7a609137d99be422a854c8300698f34d19d880531dd')
SCORER = {  # locked R1 hashes (experiments/scoring_v5/r1-lock-package.sha256)
    'scoring_v5/release_gate_v5_1.py': '5e900d6e6f48a7aa1fed17cdaa253f86451e13dedd8e0e4059553c6d09692876',
    'scoring_v5/release_gate_v5.py': '7bba5a40da44adbbc8ad0b16fb038ab2fd74f65ac73fe5e9c4c0cf6e9335918e',
    'bowl_step5/release_gate_v4.py': '1b249ae1ac1913babb7c31325898bf01823409172cb0080ccb59430c218c842c',
    'bowl_step5/release_gate_v3.py': '8a5be7d433802a9eac89bb0ec5dbe1c1592306191a3140e8f447f6c9bea15b4a',
    'bowl_step5/stability.py': '25971d4dec1ae0562c418067aca7e205f0160d115a45d7cc8475a016cce7372f',
    'bowl_step5/rim_support.py': '97d9e833bb24f0b8fd97ae6ef4cf89c5928ed97d1ce387ddfc7efd731d7a2ea9',
}
STEP5_REUSED = tuple(f'step5_training_layout{layout}_v4' for layout in range(2000, 2004))  # S2/D4, policy seed 0
COLLECT = re.compile(r'^step7_collect_layout(?P<layout>\d{4})_s(?P<seed>[01])_v1$')
STEP5 = re.compile(r'^step5_training_layout(?P<layout>200[0-3])_v4$')
APPROVED_SCHEMA = 'step7-approved-wins-v1'
EXPORT_SCHEMA = 'step7-export-v1'
ARRAYS = ('action.auxiliary_predicted_eef_9d', 'action.executed_joint_gripper', 'chunk_row', 'executed_verified',
          'fresh_model_request', 'state.eef_9d', 'state.gripper_position', 'state.joint_position',
          'video.exterior_image_1_left', 'video.wrist_image_left')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_name(case):
    """(layout, policy seed) of an allowed training case; anything else is a hard error."""
    assert not case.startswith('step8_'), f'{case}: step 8 cases never enter training (N2)'
    m = COLLECT.match(case) or STEP5.match(case)
    assert m, f'{case}: not a step 7 collection case or a named step 5 reuse case'
    layout, seed = int(m['layout']), int(m.groupdict().get('seed') or 0)
    assert layout in TRAINING_LAYOUTS, f'{case}: layout {layout} outside the A training split 2000-2031'
    return layout, seed


def explicit_cases(ledger_rows, seed1_rows=None):
    """Case set from the ledgers plus the four named step 5 cases. Returns (included, excluded) keyed by case name."""
    included, excluded = {}, {}
    for case in STEP5_REUSED:
        layout, seed = check_name(case)
        included[case] = {'layout_seed': layout, 'policy_seed': seed, 'source': 'step5 reuse (S2/D4)'}
    for rows, seed, layouts, label in ((ledger_rows, 0, range(2004, 2032), 'training-collection-v1'),
                                       (seed1_rows or [], 1, TRAINING_LAYOUTS, 'training-collection-v1-seed1')):
        stage = {}
        for row in rows:
            if 'case' not in row:
                continue
            case = row['case']
            layout, name_seed = check_name(case)
            assert COLLECT.match(case), f'{case}: ledger {label} may only hold step 7 collection cases'
            assert name_seed == seed and layout in layouts, f'{case}: wrong seed or layout for {label}'
            assert row.get('layout_seed', layout) == layout and row.get('policy_seed', seed) == seed, f'{case}: ledger fields disagree'
            prev = stage.get(case)
            assert (prev is None and row['stage'] == 'started') or (prev == 'started' and row['stage'] != 'started'), \
                f'{case}: ledger rows out of order or repeated'
            stage[case] = row['stage']
            if row['stage'] == 'finished':
                assert row.get('infrastructure_ok') is True, f'{case}: finished without infrastructure_ok'
                included[case] = {'layout_seed': layout, 'policy_seed': seed, 'source': label}
        for case, last in stage.items():
            if last != 'finished':
                excluded[case] = f'{label}: ledger stage {last!r} (not finished)'
    return included, excluded


def runner_hash(source_hashes_text):
    """The recorded hash of the step 5 episode runner, from a case's source-hashes.txt."""
    found = [line.split()[0] for line in source_hashes_text.splitlines()
             if line.strip() and line.split()[-1].endswith('/' + RUNNER[0])]
    assert len(found) == 1, 'source-hashes.txt must record the episode runner exactly once'
    return found[0]


def load_approved(approved, cases):
    """(ok, rejected, pending): {case: entry} for OK, {case: reason} for REJECT, {case: entry} for PENDING. Malformed entries are errors."""
    assert approved.get('schema') == APPROVED_SCHEMA, 'approved-wins schema'
    assert approved.get('scorer_sha256') == SCORER['scoring_v5/release_gate_v5_1.py'], 'approved wins must name the locked v5.1'
    ok, rejected, pending = {}, {}, {}
    for entry in approved['episodes']:
        case = entry['case']
        check_name(case)
        assert case in cases, f'{case}: approved but not in the explicit case set (finished ledger rows + named step 5 cases)'
        assert case not in ok and case not in rejected and case not in pending, f'{case}: approved twice'
        assert isinstance(entry['win_control'], int) and entry['win_control'] >= 1, f'{case}: win_control'
        assert entry.get('reviewer'), f'{case}: reviewer'
        assert entry['verdict'] in ('OK', 'REJECT', 'PENDING'), f'{case}: verdict {entry["verdict"]!r}'
        if entry['verdict'] == 'OK':
            ok[case] = entry
        elif entry['verdict'] == 'REJECT':
            rejected[case] = 'reviewer verdict REJECT'
        else:
            pending[case] = entry
    return ok, rejected, pending


def verdict_gaps(passes, ok, rejected):
    """C-a: every v5.1 pass in the case set needs a final verdict (OK or REJECT); returns the passes that lack one, sorted.
    A missing or PENDING pass would let a win be dropped by omission, which also moves the D3 count of 6."""
    return sorted(set(passes) - set(ok) - set(rejected))


def require_finished(rows, mode, label):
    """C-b: export reads only complete batches; the last ledger row must be batch_finished for this mode."""
    assert rows and rows[-1].get('stage') == 'batch_finished' and rows[-1].get('mode') == mode, \
        f'{label}: ledger does not end with batch_finished (mode {mode}); export waits for the whole batch'


def win_row(steps, win_control):
    """Row index of a control in a v4 trace whose steps are exactly 1..n; rows 0..that index are kept."""
    steps = list(steps)
    assert steps == list(range(1, len(steps) + 1)), 'trace steps must be 1..n'
    row = steps.index(win_control)
    assert row == win_control - 1
    return row


def shift_errors(a, b, shifts=(-2, -1, 0, 1, 2)):
    """Max |a[i] - b[i + s]| per shift: the row alignment holds when only shift 0 is ~0."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {}
    for s in shifts:
        x, y = (a[:len(a) - s], b[s:]) if s >= 0 else (a[-s:], b[:len(b) + s])
        out[s] = float(np.abs(x - y).max())
    return out


def check_scorer(base):
    for relative, expected in SCORER.items():
        assert sha256(base / relative) == expected, f'{relative}: not the locked hash'


def import_scorer(base):
    check_scorer(base)
    sys.path[:0] = [str(base / 'scoring_v5'), str(base / 'bowl_step5')]
    from release_gate_v5_1 import score, review_flags
    return score, review_flags


def case_paths(root, case):
    out = root / 'RoboLab/output' / case / TASK
    return {'hdf5': out / 'synchronized.hdf5', 'metric': out / 'strict-metric.json',
            'hashes': root / 'runs' / case / 'source-hashes.txt', 'integrity': root / 'runs' / case / 'capture-integrity.json'}


def read_ledgers(args, finished=False):
    rows = [json.loads(x) for x in Path(args.ledger).read_text().splitlines()]
    seed1 = [json.loads(x) for x in Path(args.seed1_ledger).read_text().splitlines()] if args.seed1_ledger else None
    if finished:  # C-b, export only; candidates still reads snapshots of running batches
        require_finished(rows, 'first', 'training-collection-v1')
        if seed1 is not None:
            require_finished(seed1, 'seed1', 'training-collection-v1-seed1')
    return explicit_cases(rows, seed1)


def identify(root, case, info):
    """Shared per-case identity checks; returns the metric."""
    p = case_paths(root, case)
    assert runner_hash(p['hashes'].read_text()) == RUNNER[1], f'{case}: not the approved step 5 runner (S2)'
    integrity = json.loads(p['integrity'].read_text())
    metric = json.loads(p['metric'].read_text())
    for record in (integrity, metric):
        assert record['layout_seed'] == info['layout_seed'] and record['policy_seed'] == info['policy_seed'], case
        assert record['task_class'] == [TASK], case
    assert metric['schema'] == 'command-linked-supported-release-v4', case
    assert integrity['trace_rows'] == len(metric['trace']), f'{case}: capture-integrity trace_rows != metric trace length (C-c)'
    return metric


def candidates(args):
    root = Path(args.root)
    score, review_flags = import_scorer(root / 'experiments')
    included, excluded = read_ledgers(args)
    out = []
    for case in sorted(included):
        metric = identify(root, case, included[case])
        s = score(case_paths(root, case)['metric'])
        v4 = metric['stable_success_s'] is not None
        out.append({'case': case, **included[case], 'v5_1_pass': s['pass'], 'win_control': s['success_control'],
                    'rows': s['rows'], 'cut_rows': s['rows'] - s['success_control'] if s['pass'] else None,
                    'live_v4_pass': v4, 'fail_reason': s['fail_reason'], 'review_flags': review_flags(s, v4)})
    template = {'schema': APPROVED_SCHEMA, 'scorer_sha256': SCORER['scoring_v5/release_gate_v5_1.py'],
                'note': 'Reviewer sets verdict OK or REJECT per v5.1 pass; only OK entries are converted.',
                'episodes': [{'case': c['case'], 'win_control': c['win_control'], 'verdict': 'PENDING', 'reviewer': 'shadow-reviewer'}
                             for c in out if c['v5_1_pass']]}
    result = {'cases': out, 'excluded_before_scoring': excluded, 'approved_wins_template': template}
    with Path(args.out).open('x') as stream:
        stream.write(json.dumps(result, indent=1) + '\n')
    for c in out:
        print(f"{c['case']:34} v5.1={'pass' if c['v5_1_pass'] else 'fail':4} win={c['win_control']!s:4} rows={c['rows']:3} "
              f"live_v4={c['live_v4_pass']!s:5} flags={', '.join(c['review_flags']) or '-'}")
    for case, why in excluded.items():
        print(f'{case:34} excluded: {why}')


def export(args):
    import h5py
    import numpy as np
    root, out = Path(args.root), Path(args.out)
    score, _ = import_scorer(root / 'experiments')
    included, excluded = read_ledgers(args, finished=True)
    approved_path = Path(args.approved)
    ok, rejected, pending = load_approved(json.loads(approved_path.read_text()), included)
    passes = set()
    for case in sorted(included):
        identify(root, case, included[case])
        if score(case_paths(root, case)['metric'])['pass']:
            passes.add(case)
    gaps = verdict_gaps(passes, ok, rejected)
    assert not gaps, f'every v5.1 pass needs a verdict OK or REJECT before export (C-a); no verdict: {gaps}'
    out.mkdir(parents=True, exist_ok=False)
    episodes = []
    for index, case in enumerate(sorted(ok)):
        info, entry, p = included[case], ok[case], case_paths(root, case)
        metric = identify(root, case, info)
        s = score(p['metric'])
        assert s['pass'] and s['success_control'] == entry['win_control'], f'{case}: locked v5.1 does not reproduce the approved win'
        trace = metric['trace']
        row = win_row([r['step'] for r in trace], entry['win_control'])
        with h5py.File(p['hdf5'], 'r') as f:
            attrs = {k: v.item() if isinstance(v, np.generic) else v for k, v in f.attrs.items()}
            assert attrs['complete'] and attrs['schema'] == 'step5-pre-action-controlled-release-v4', case
            assert attrs['layout_seed'] == info['layout_seed'] and attrs['policy_seed'] == info['policy_seed'], case
            assert attrs['registered_environment'] == metric['registered_environment'] and attrs['task_class'] == TASK, case
            assert sorted(f) == sorted(ARRAYS), f'{case}: unexpected HDF5 keys {sorted(f)}'
            arrays = {k: f[k][:] for k in ARRAYS}
        n = len(trace)
        assert all(len(a) == n for a in arrays.values()), f'{case}: HDF5 rows != trace rows'
        assert arrays['executed_verified'].all(), case
        command = np.array([r['command'] for r in trace])
        closure = np.array([r['pre']['closure'] for r in trace])
        np.testing.assert_array_equal(arrays['action.executed_joint_gripper'][:, -1], command)
        grip = arrays['state.gripper_position'][:, 0]
        shifts = shift_errors(grip, closure)
        assert shifts[0] <= 1e-6, f'{case}: gripper state does not match the trace closure row for row'
        # proof at the cut: the kept last row is the trace row of the win control
        assert trace[row]['step'] == entry['win_control']
        assert abs(float(grip[row]) - trace[row]['pre']['closure']) <= 1e-6
        assert float(arrays['action.executed_joint_gripper'][row, -1]) == trace[row]['command']
        kept = row + 1
        cut = {k: a[:kept] for k, a in arrays.items()}
        for k in ('state.eef_9d', 'state.gripper_position', 'state.joint_position',
                  'action.auxiliary_predicted_eef_9d', 'action.executed_joint_gripper'):
            assert np.isfinite(cut[k]).all(), f'{case}: non-finite {k}'
        assert kept >= 40, f'{case}: {kept} rows is shorter than one 40-row action window'
        meta = {'case': case, 'episode_index': index, **info, 'win_control': entry['win_control'], 'win_row': row,
                'original_rows': n, 'kept_rows': kept, 'cut': kept < n, 'live_v4_stable_success_s': metric['stable_success_s'],
                'reviewer': entry['reviewer'], 'source_hdf5_sha256': sha256(p['hdf5']), 'source_metric_sha256': sha256(p['metric']),
                'gripper_vs_trace_closure_by_shift': shifts, 'instruction': attrs['instruction'], 'step_dt': attrs['step_dt'],
                'capture_metadata': attrs}
        destination = out / f'{case}.npz'
        with destination.open('xb') as stream:
            np.savez_compressed(stream, **cut, trace_command=command[:kept], trace_pre_closure=closure[:kept],
                                trace_step=np.array([r['step'] for r in trace[:kept]]), metadata_json=np.asarray(json.dumps(meta)))
        meta['npz_sha256'] = sha256(destination)
        episodes.append(meta)
        print(f"{case}: rows {n} -> {kept} (win control {entry['win_control']}, {'cut' if kept < n else 'not cut'})", flush=True)
    not_approved = {c: 'v5.1 fail' for c in included if c not in ok and c not in rejected}
    manifest = {'schema': EXPORT_SCHEMA, 'approved_wins': str(approved_path), 'approved_wins_sha256': sha256(approved_path),
                'scorer_sha256': SCORER, 'runner_sha256': RUNNER[1], 'converter_sha256': sha256(__file__),
                'episodes': episodes, 'excluded': {**excluded, **rejected, **not_approved}}
    with (out / 'export-manifest.json').open('x') as stream:
        stream.write(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({'episodes': len(episodes), 'frames': sum(e['kept_rows'] for e in episodes),
                      'excluded': len(manifest['excluded'])}))


def build(args):
    import subprocess
    import numpy as np
    import pandas as pd
    source, output = Path(args.export), Path(args.out)
    manifest = json.loads((source / 'export-manifest.json').read_text())
    assert manifest['schema'] == EXPORT_SCHEMA and manifest['episodes'], 'empty or foreign export'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'meta').mkdir()
    fps, instruction, total, lengths = 15, None, 0, []
    for e in manifest['episodes']:
        path = source / f"{e['case']}.npz"
        assert sha256(path) == e['npz_sha256'], f"{e['case']}: npz changed since export"
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data['metadata_json']))
        assert meta['case'] == e['case'] and meta['episode_index'] == e['episode_index']
        assert round(1 / float(meta['step_dt'])) == fps
        instruction = instruction or meta['instruction']
        assert meta['instruction'] == instruction, 'one task only'
        i, n = e['episode_index'], e['kept_rows']
        assert i == len(lengths), 'episode indices must be dense and ordered'
        state = np.concatenate([data['state.eef_9d'], data['state.gripper_position'], data['state.joint_position']], axis=1)
        executed = data['action.executed_joint_gripper']
        action = np.concatenate([data['action.auxiliary_predicted_eef_9d'], executed[:, 7:8], executed[:, :7]], axis=1)
        assert state.shape == action.shape == (n, 17) and np.isfinite(state).all() and np.isfinite(action).all()
        frame = pd.DataFrame({'observation.state': list(state), 'action': list(action),
                              'timestamp': np.arange(n, dtype=np.float64) / fps, 'frame_index': np.arange(n),
                              'episode_index': np.full(n, i, dtype=np.int64), 'index': total + np.arange(n),
                              'task_index': np.zeros(n, dtype=np.int64)})
        table = output / f'data/chunk-000/episode_{i:06d}.parquet'
        table.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(table, index=False)
        for camera in ('exterior_image_1_left', 'wrist_image_left'):
            frames = data[f'video.{camera}']
            assert frames.shape == (n, 180, 320, 3) and frames.dtype == np.uint8
            destination = output / f'videos/chunk-000/observation.images.{camera}/episode_{i:06d}.mp4'
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '320x180', '-r', str(fps),
                            '-i', 'pipe:0', '-an', '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'fast', str(destination)],
                           input=frames.tobytes(), check=True)
        lengths.append(n)
        total += n
    groups = [('eef_9d', 0, 9), ('gripper_position', 9, 10), ('joint_position', 10, 17)]
    modality = {m: {k: {'start': a, 'end': b} for k, a, b in groups} for m in ('state', 'action')}
    modality['video'] = {c: {'original_key': f'observation.images.{c}'} for c in ('exterior_image_1_left', 'wrist_image_left')}
    modality['annotation'] = {'language.language_instruction': {'original_key': 'task_index'}}
    features = {'observation.state': {'dtype': 'float32', 'shape': [17]}, 'action': {'dtype': 'float32', 'shape': [17]}}
    for c in ('exterior_image_1_left', 'wrist_image_left'):
        features[f'observation.images.{c}'] = {'dtype': 'video', 'shape': [180, 320, 3], 'names': ['height', 'width', 'channels'],
                                               'video_info': {'video.fps': fps, 'video.codec': 'h264', 'video.pix_fmt': 'gbrp'}}
    info = {'codebase_version': 'v2.0', 'robot_type': 'droid_franka_robotiq_sim', 'total_episodes': len(lengths),
            'total_frames': total, 'total_tasks': 1, 'total_videos': 2 * len(lengths), 'total_chunks': 1, 'chunks_size': 1000,
            'fps': fps, 'splits': {'train': f'0:{len(lengths)}'},
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4', 'features': features}
    for name, value in (('info', info), ('modality', modality)):
        (output / f'meta/{name}.json').write_text(json.dumps(value, indent=2) + '\n')
    (output / 'meta/tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': instruction}) + '\n')
    (output / 'meta/episodes.jsonl').write_text(''.join(json.dumps({'episode_index': i, 'tasks': [instruction], 'length': n}) + '\n'
                                                        for i, n in enumerate(lengths)))
    (output / 'PROVENANCE.json').write_text(json.dumps({
        'export': str(source), 'export_manifest_sha256': sha256(source / 'export-manifest.json'),
        'converter_sha256': sha256(__file__), 'episodes': [{k: e[k] for k in ('case', 'episode_index', 'layout_seed', 'policy_seed',
                                                                              'win_control', 'original_rows', 'kept_rows', 'cut', 'npz_sha256')}
                                                           for e in manifest['episodes']],
        'excluded': manifest['excluded'],
        'purpose': 'step 7 P0 -> P1 success-filtered self-demonstration on Robot A (STEP7_PLAN); reviewed v5.1 wins only',
        'state': 'pre-action model-format state', 'joint_gripper_labels': 'actual executed absolute targets',
        'eef_labels': 'auxiliary absolute P0 prediction, not independently verified controller targets',
        'video': 'pre-action RGB resized by stock client; lossless RGB H264', 'cut': 'rows 0..win row (control S keeps S rows)',
    }, indent=1) + '\n')
    sys.path.insert(0, str(ROOT / 'Isaac-GR00T'))
    from gr00t.data.stats import main as stats_main
    from gr00t.data.types import EmbodimentTag
    stats_main(dataset_path=output, embodiment_tag=EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
               modality_config_path=str(ROOT / 'experiments/bowl_step5/checkpoint_modalities.py'))
    print(json.dumps({'output': str(output), 'episodes': len(lengths), 'frames': total, 'lengths': lengths}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('candidates', 'export'):
        p = sub.add_parser(name)
        p.add_argument('--root', default=str(ROOT))
        p.add_argument('--ledger', required=True, help='runs/training-collection-v1.jsonl')
        p.add_argument('--seed1-ledger', help='runs/training-collection-v1-seed1.jsonl (D3 only)')
        p.add_argument('--out', required=True)
        if name == 'export':
            p.add_argument('--approved', required=True)
    p = sub.add_parser('build')
    p.add_argument('--export', required=True)
    p.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    {'candidates': candidates, 'export': export, 'build': build}[args.command](args)


if __name__ == '__main__':
    main()
