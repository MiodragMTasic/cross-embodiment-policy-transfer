"""Read-only A/B USD joint graph export; no physics, policy, or asset mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--case',required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
assert args.case.startswith('step6_kinematics_') and all(c.isalnum() or c=='_' for c in args.case)
root=Path.home()/'robot-transfer/bowl'
output=root/'runs'/args.case
output.mkdir(parents=True,exist_ok=False)
app=AppLauncher(args).app

try:
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics

    def quat(q):
        return [float(q.GetReal()),*[float(x) for x in q.GetImaginary()]]

    def number(x):
        if x is None:return None
        return float(x) if np.isfinite(x) else None

    def export(label,relative,root_name,tool_name,expected):
        asset=root/'RoboLab/assets/robots'/relative
        stage=Usd.Stage.Open(str(asset))
        assert stage and UsdGeom.GetStageMetersPerUnit(stage)==1., 'Expected meter-scaled asset'
        xcache=UsdGeom.XformCache(Usd.TimeCode.Default())
        body_prims=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        bodies={}
        for p in body_prims:
            m=np.array(xcache.GetLocalToWorldTransform(p))
            assert np.isfinite(m).all()
            scale=np.linalg.svd(m[:3,:3],compute_uv=False)
            assert np.allclose(scale,1,atol=1e-5,rtol=0), (str(p.GetPath()),scale)
            bodies[str(p.GetPath())]={'name':p.GetName(),'world_row_matrix':m.tolist()}
        roots=[p for p,v in bodies.items() if v['name']==root_name]
        tips=[p for p,v in bodies.items() if v['name']==tool_name and p not in roots]
        assert len(roots)==len(tips)==1,(label,roots,tips)
        base,tool=roots[0],tips[0]
        joints=[]; graph={p:[] for p in bodies}
        for p in stage.Traverse():
            if not p.IsA(UsdPhysics.Joint):continue
            j=UsdPhysics.Joint(p)
            b0=[str(x) for x in j.GetBody0Rel().GetTargets()]
            b1=[str(x) for x in j.GetBody1Rel().GetTargets()]
            assert len(b0)<=1 and len(b1)<=1,(p,b0,b1)
            e={'name':p.GetName(),'path':str(p.GetPath()),'type':p.GetTypeName(),
                'body0':b0[0] if b0 else None,'body1':b1[0] if b1 else None,
                'enabled':bool(j.GetJointEnabledAttr().Get()),
                'exclude_from_articulation':j.GetExcludeFromArticulationAttr().Get(),
                'local_pos0':[float(x) for x in j.GetLocalPos0Attr().Get()],
                'local_rot0_wxyz':quat(j.GetLocalRot0Attr().Get()),
                'local_pos1':[float(x) for x in j.GetLocalPos1Attr().Get()],
                'local_rot1_wxyz':quat(j.GetLocalRot1Attr().Get())}
            if p.IsA(UsdPhysics.RevoluteJoint):
                r=UsdPhysics.RevoluteJoint(p)
                e.update(axis=str(r.GetAxisAttr().Get()),lower_limit_degrees=number(r.GetLowerLimitAttr().Get()),
                    upper_limit_degrees=number(r.GetUpperLimitAttr().Get()))
            index=len(joints);joints.append(e)
            if e['enabled'] and e['body0'] in graph and e['body1'] in graph:
                graph[e['body0']].append((e['body1'],index,1))
                graph[e['body1']].append((e['body0'],index,-1))
        paths=[]
        def walk(body,seen,chain):
            if body==tool:
                paths.append(chain);return
            for other,index,direction in graph[body]:
                if other not in seen:
                    walk(other,seen|{other},chain+[{'joint_index':index,'direction':direction}])
            assert len(paths)<=1,'Ambiguous base-to-tool chain'
        walk(base,{base},[])
        assert len(paths)==1,(label,base,tool)
        chain=paths[0]
        actuated=[joints[e['joint_index']]['name'] for e in chain if joints[e['joint_index']]['type']=='PhysicsRevoluteJoint']
        assert actuated==expected,(label,actuated)
        assert all(joints[e['joint_index']]['type'] in ('PhysicsRevoluteJoint','PhysicsFixedJoint') for e in chain)
        layers=[]
        for layer in stage.GetUsedLayers():
            path=Path(layer.realPath) if layer.realPath else None
            if path and path.is_file():
                layers.append({'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return {'label':label,'asset':str(asset),'asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
            'meters_per_unit':1.,'base_body':base,'raw_tool_body':tool,'bodies':bodies,'joints':joints,
            'chain':chain,'arm_joint_names':expected,'used_layers':layers,
            'convention':'USD frame positions in meters, quaternion wxyz, row matrices only for saved static world transforms. FK uses column matrices T0 R_axis(q) inverse(T1) for body0->body1; reversed edge uses inverse. Runtime joint angles must be radians.',
            'status':'Static export only; FK not yet validated against live simulator states.'}

    result={'A':export('A','franka_robotiq_2f_85_flattened.usd','panda_link0','base_link',[f'panda_joint{i}' for i in range(1,8)]),
            'B':export('B','kinova_gen3_robotiq_2f85/kinova_gen3_7dof_robotiq_2f85.usd','base_link','robotiq_85_base_link',[f'joint_{i}' for i in range(1,8)]),
            'complete':True}
    (output/'joint_graphs.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('STATIC_KINEMATICS_EXPORT_COMPLETE',flush=True)
except BaseException:
    error=traceback.format_exc();(output/'failure.txt').write_text(error);print(error,flush=True)
    raise
finally:
    app.close()
