"""Read pinned source geometry only; make no changes to the scene."""
import json
from pathlib import Path
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
try:
    from pxr import Usd, UsdGeom
    import numpy as np
    root = Path.home() / 'robot-transfer/bowl'
    stage = Usd.Stage.Open(str(root / 'RoboLab/assets/scenes/banana_bowl.usda'))
    cache = UsdGeom.XformCache()
    report = {'source': 'RoboLab/assets/scenes/banana_bowl.usda', 'objects': {}}
    for name in ['banana', 'bowl']:
        candidates = [p for p in stage.Traverse() if p.GetName() == name]
        assert len(candidates) == 1, [(str(p.GetPath())) for p in candidates]
        obj = candidates[0]
        record = {'path': str(obj.GetPath()), 'meshes': []}
        for p in Usd.PrimRange(obj):
            if not p.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(p)
            xf, _ = cache.ComputeRelativeTransform(p, obj)
            points = np.asarray([tuple(xf.Transform(pt)) for pt in mesh.GetPointsAttr().Get()])
            if len(points):
                record['meshes'].append({'path': str(p.GetPath()), 'points': points.tolist(),
                    'min': points.min(axis=0).tolist(), 'max': points.max(axis=0).tolist(),
                    'collision_enabled': p.GetAttribute('physics:collisionEnabled').Get(),
                    'purpose': str(mesh.GetPurposeAttr().Get())})
        report['objects'][name] = record
    destination = root / 'experiments/bowl_step5/geometry-audit.json'
    destination.write_text(json.dumps(report))
    for name, rec in report['objects'].items():
        print('GEOMETRY', name, [{k:v for k,v in m.items() if k!='points'} for m in rec['meshes']], flush=True)
finally:
    app.close()
