"""Register the exact P0 post-training observation/action contract."""
import json
from pathlib import Path
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import ModalityConfig

ROOT = Path.home() / 'robot-transfer/bowl'
manifest = json.loads((ROOT / 'models/model-manifest.json').read_text())
p0 = next(m['snapshot'] for m in manifest['models'] if m['repo'] == 'nvidia/GR00T-N1.7-DROID')
processor = json.loads((Path(p0) / 'processor_config.json').read_text())['processor_kwargs']
for embodiment, configs in processor['modality_configs'].items():
    MODALITY_CONFIGS[embodiment] = {key: ModalityConfig(**value) for key, value in configs.items()}
assert MODALITY_CONFIGS['oxe_droid_relative_eef_relative_joint']['video'].delta_indices == [0]
