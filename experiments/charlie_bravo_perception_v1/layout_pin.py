"""Integrity of the frozen scene input, not a perception input."""
import hashlib
from pathlib import Path
RELATIVE=Path('RoboLab/output/step8_A_P0_l4003_s0_v1/BananaInBowlUniformInitPose10cmTask/strict-metric.json')
EXPECTED='6143585781b7115570a8b147b9afc36508be0ad733cd34a78fac5e290c3c8f83'
def verify_layout(root):
    path=Path(root)/RELATIVE
    if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED:
        raise ValueError('frozen layout input mismatch')
    return path
