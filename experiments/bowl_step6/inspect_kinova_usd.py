"""Read-only static USD inspection. No environment, physics stepping or asset writes."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
root=Path.home()/'robot-transfer/bowl'
output=root/'runs/step6_B_usd_inspection_v1'
output.mkdir(parents=True,exist_ok=False)
app=AppLauncher(args).app
try:
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics
    asset=root/'RoboLab/assets/robots/kinova_gen3_robotiq_2f85/kinova_gen3_7dof_robotiq_2f85.usd'
    stage=Usd.Stage.Open(str(asset))
    assert stage
    caches={
        'original_purposes':UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']),
        'all_purposes':UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'])}
    result={'asset':str(asset),'asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
            'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'default_prim':str(stage.GetDefaultPrim().GetPath()),'tips':{}}
    for side in ('left','right'):
        name=f'robotiq_85_{side}_finger_tip_link'
        bodies=[p for p in stage.Traverse() if p.GetName()==name]
        assert len(bodies)==1,(name,[str(p.GetPath()) for p in bodies])
        body=bodies[0]; entries=[]
        for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
            imageable=UsdGeom.Imageable(prim)
            e={'path':str(prim.GetPath()),'type':prim.GetTypeName(),'active':prim.IsActive(),
               'instance':prim.IsInstance(),'instanceable':prim.IsInstanceable(),'proxy':prim.IsInstanceProxy(),
               'loaded':prim.IsLoaded(),'authored_references':str(prim.GetMetadata('references')),
               'applied_schemas':prim.GetAppliedSchemas(),'collision_api':prim.HasAPI(UsdPhysics.CollisionAPI),
               'purpose':str(imageable.ComputePurpose()) if imageable else None,
               'collision_enabled':UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() if prim.HasAPI(UsdPhysics.CollisionAPI) else None,
               'bounds':{}}
            for label,cache in caches.items():
                bound=cache.ComputeRelativeBound(prim,body).ComputeAlignedRange()
                lo,hi=np.array(bound.GetMin()),np.array(bound.GetMax())
                e['bounds'][label]={'min':lo.tolist(),'max':hi.tolist(),'nonempty':bool(np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(hi>lo))}
            for attr_name in ('extent','points','xformOp:scale'):
                value=prim.GetAttribute(attr_name).Get()
                if value is not None:
                    a=np.asarray(value)
                    e[attr_name]={'shape':list(a.shape),'min':a.min(axis=0).tolist(),'max':a.max(axis=0).tolist()}
            entries.append(e)
        result['tips'][side]={'body_path':str(body.GetPath()),'descendants':entries}
    result['complete']=True
    (output/'inspection.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('STATIC_USD_INSPECTION_COMPLETE',flush=True)
except BaseException:
    error=traceback.format_exc();(output/'failure.txt').write_text(error);print(error,flush=True)
    raise
finally:
    app.close()
