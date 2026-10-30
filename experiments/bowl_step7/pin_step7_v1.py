"""Pin step 7 checkpoints by content (STEP7_PLAN_2026-10-23.md §4a, steps 1 and 2). Never overwrites.

python3 pin_step7_v1.py candidates <training output dir> <candidates-sha256.json>
    Hashes every file of checkpoint-{500,1000,1500,2000} (all four must exist) into one schema step7-candidates-sha256-v1 file.
python3 pin_step7_v1.py select <candidates-sha256.json> <select-step7.json> <c500|c1000|c1500|c2000> <P1-manifest.json>
    Requires the adjudicated D2 choice in select-step7.json (score_select_step7.py --adjudicated) to name that candidate.
    Re-hashes it against its pin and writes its entry in the P1-manifest schema that run_step8.sh reads
    (experiments/bowl_step8/STEP8_RUNNERS.md): {"checkpoint_dir", "files": [{"path", "size", "sha256"}]}. Provenance
    (choice, label, hashes of both inputs) goes to a sidecar <P1-manifest>.provenance.json, never into the manifest.
"""
import hashlib
import json
from pathlib import Path
import sys

STEPS = (500, 1000, 1500, 2000)


def digest(d):
    files = []
    for p in sorted(x for x in d.rglob('*') if x.is_file()):
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(b)
        files.append({'path': str(p.relative_to(d)), 'size': p.stat().st_size, 'sha256': h.hexdigest()})
    assert files, d
    return files


def write_new(path, data):
    path = Path(path)
    assert not path.exists(), f'never overwrite {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + '\n')
    print(path, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)


mode, *args = sys.argv[1:]
if mode == 'candidates':
    out_dir, target = Path(args[0]).resolve(), args[1]
    present = sorted(p.name for p in out_dir.glob('checkpoint-*'))
    assert present == sorted(f'checkpoint-{s}' for s in STEPS), present
    write_new(target, {'schema': 'step7-candidates-sha256-v1', 'training_output': str(out_dir),
                       'candidates': {f'c{s}': {'checkpoint_dir': str(out_dir / f'checkpoint-{s}'),
                                                'files': digest(out_dir / f'checkpoint-{s}')} for s in STEPS}})
elif mode == 'select':
    pins_path, select_path, name, target = args
    pins, sel = json.loads(Path(pins_path).read_text()), json.loads(Path(select_path).read_text())
    assert pins['schema'] == 'step7-candidates-sha256-v1' and sel['schema'] == 'step7-select-v1'
    assert sel.get('adjudicated_from') and sel['choice']['final'], 'only the final adjudicated choice (S3, D2)'
    assert sel['choice']['P1'] == name, f"adjudicated choice is {sel['choice']['P1']}, not {name}"
    c = pins['candidates'][name]
    assert digest(Path(c['checkpoint_dir'])) == c['files'], f'{name} no longer matches its pin'
    write_new(target, {'checkpoint_dir': c['checkpoint_dir'], 'files': c['files']})
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    write_new(target + '.provenance.json', {'candidate': name, 'choice': sel['choice'], 'notes': sel['notes'],
                                            'candidates_sha256': sha(pins_path), 'select_step7_sha256': sha(select_path)})
else:
    raise SystemExit(__doc__)
