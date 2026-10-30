"""Use RoboLab's existing h5py to bridge into GR00T without changing either env."""
import sys
from pathlib import Path
import h5py
import numpy as np

source, destination = map(Path, sys.argv[1:])
assert not destination.exists()
with h5py.File(source, 'r') as f:
    assert f.attrs['complete']
    arrays = {key: f[key][:] for key in f}
    arrays.update({key: np.asarray(f.attrs[key]) for key in ('complete', 'step_dt', 'instruction')})
    np.savez_compressed(destination, **arrays)
print(destination)
