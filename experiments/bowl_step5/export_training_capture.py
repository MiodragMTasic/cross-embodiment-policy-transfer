"""Export a complete, identified training-split capture; never overwrite evidence."""
import hashlib,json,sys
from pathlib import Path
import h5py
import numpy as np
source,destination=map(Path,sys.argv[1:]); assert not destination.exists()
assert destination.suffix == '.npz', 'Use an explicit .npz output path'
metric_path=source.with_name('strict-metric.json');metric=json.loads(metric_path.read_text())
assert metric['schema']=='command-linked-supported-release-v4'
assert 2000<=metric['layout_seed']<=2031
assert source.parent.parent.name==f"step5_training_layout{metric['layout_seed']}_v4"
with h5py.File(source,'r') as f:
    assert f.attrs['complete']
    attrs={k:v.item() if isinstance(v,np.generic) else v for k,v in f.attrs.items()}
    assert attrs['schema']=='step5-pre-action-controlled-release-v4'
    assert attrs['layout_seed']==metric['layout_seed'] and attrs['policy_seed']==metric['policy_seed']
    assert attrs['task_class']==','.join(metric['task_class'])
    assert attrs['registered_environment']==metric['registered_environment']
    arrays={k:f[k][:] for k in f}
    assert all(len(a)==len(metric['trace']) for a in arrays.values())
    assert arrays['executed_verified'].all()
    np.testing.assert_array_equal(arrays['action.executed_joint_gripper'][:,-1],[r['command'] for r in metric['trace']])
    arrays.update({k:np.asarray(attrs[k]) for k in ('complete','step_dt','instruction')})
    arrays.update(source_case_name=np.asarray(source.parent.parent.name),
                  capture_metadata_json=np.asarray(json.dumps(attrs)),
                  source_hdf5_sha256=np.asarray(hashlib.sha256(source.read_bytes()).hexdigest()),
                  source_metric_sha256=np.asarray(hashlib.sha256(metric_path.read_bytes()).hexdigest()))
    with destination.open('xb') as output:
        np.savez_compressed(output,**arrays)
print(json.dumps({'source':str(source),'output':str(destination),'frames':len(metric['trace']),
                  'npz_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}))
