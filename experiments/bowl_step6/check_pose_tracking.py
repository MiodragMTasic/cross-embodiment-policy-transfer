"""Bounded raw-tool tracking and measured gripper/camera inventory; no policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path.home() / 'robot-transfer/bowl'
REPO = ROOT / 'RoboLab'
sys.path.insert(0, str(REPO))
os.chdir(REPO)
import cv2
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--case', required=True)
parser.add_argument('--robot', choices=['A', 'B'], required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert args.case.startswith(f'step6_{args.robot}_tracking_') and all(c.isalnum() or c == '_' for c in args.case)
output = ROOT / 'runs' / args.case
output.mkdir(parents=True, exist_ok=False)
args.enable_cameras = True
app = AppLauncher(args).app

PHASES = [('settle',45), ('x',30), ('y',30), ('z',30), ('rotate',30),
          ('return',45), ('close',45), ('reopen',45)]
env = video = None
rows = []
complete = False
try:
    import numpy as np
    import torch
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.utils.stage import get_current_stage
    from robolab.constants import set_output_dir
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
    from robolab.core.utils.video_utils import VideoWriter
    from robolab.robots.kinova_gen3 import _to_torch, GRIPPER_JOINT_COMMANDS
    from rigid_chain import RigidChain, pose_matrix, rigid_inverse, axis_rotation
    from local_ik import solve, physical_limits, pose_error

    def arr(x):
        return _to_torch(x).detach().cpu().numpy().copy()

    def pose(x):
        return pose_matrix(x[:3], x[3:7])

    def colliders(stage, path):
        body = stage.GetPrimAtPath(path)
        assert body.IsValid() and body.HasAPI(UsdPhysics.RigidBodyAPI), path
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        result = []
        seen = set()
        for declaration in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not declaration.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(declaration).GetCollisionEnabledAttr().Get():
                continue
            if declaration.IsA(UsdGeom.Mesh):
                meshes = [declaration]
            else:
                # A's authored CollisionAPI is on Xform wrappers. Inventory their mesh geometry;
                # do NOT assert that PhysicsCollisionAPI is inherited or that these are cooked shapes.
                assert declaration.IsA(UsdGeom.Xform), str(declaration.GetPath())
                descendants = list(Usd.PrimRange(declaration, Usd.TraverseInstanceProxies()))
                assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in descendants), 'Nested rigid body'
                meshes = [p for p in descendants if p.IsA(UsdGeom.Mesh)]
                assert meshes, str(declaration.GetPath())
            for prim in meshes:
                key = str(prim.GetPath())
                assert key not in seen, 'Ambiguous duplicate geometry declaration: '+key
                seen.add(key)
                mesh = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                assert mesh.ndim == 2 and mesh.shape[1] == 3 and len(mesh) and np.isfinite(mesh).all()
                # Only mesh-to-body is obtained from USD. Runtime body poses below use physics buffers.
                transform = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()
                vertices = np.c_[mesh,np.ones(len(mesh))] @ np.asarray(transform)
                independent = np.array([transform.Transform(Gf.Vec3d(*v)) for v in mesh])
                assert np.allclose(vertices[:,3],1) and np.allclose(vertices[:,:3],independent,atol=1e-10)
                result.append({'path':key, 'purpose':str(UsdGeom.Imageable(prim).ComputePurpose()),
                    'collision_declaration_path':str(declaration.GetPath()),'declaration_type':declaration.GetTypeName(),
                    'mesh_has_own_collision_api':bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
                    'geometry_scope':'Authored mesh geometry associated with a collision declaration; not proof of cooked/runtime collision shape.',
                    'mesh_to_body_row_matrix':np.asarray(transform).tolist(),'body_vertices':vertices[:,:3].tolist()})
        assert result, path
        return result

    set_output_dir(str(output))
    if args.robot == 'A':
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        auto_register_droid_envs(task='BananaInBowlTask')
        tip_names = ['left_inner_finger','right_inner_finger']
        arm_term = 'body'
        grip_targets = {'finger_joint':np.pi/4}
    else:
        from robolab.registrations.kinova.auto_env_registrations_jointpos import auto_register_kinova_envs
        auto_register_kinova_envs(task='BananaInBowlTask')
        tip_names = ['robotiq_85_left_finger_tip_link','robotiq_85_right_finger_tip_link']
        arm_term = 'arm'
        grip_targets = GRIPPER_JOINT_COMMANDS
    names = get_envs(task='BananaInBowlTask')
    assert len(names) == 1, names
    cfg = parse_env_cfg(names[0], device=args.device, seed=0, num_envs=1, use_fabric=True)
    # CameraData poses are initialization-only by default, even when rendered RGB is current.
    for name in ('wrist_cam','over_shoulder_left_camera'):
        getattr(cfg.scene,name).update_latest_camera_pose = True
    env,cfg = create_env(cfg)
    obs,_ = env.reset()
    robot = env.scene['robot']
    graph_path = ROOT/'assets/report/evidence/2026-10-21/bowl-step6/kinematics_export_v1/joint_graphs.json'
    graph = json.loads(graph_path.read_text())[args.robot]
    chain = RigidChain(graph)
    limits = physical_limits(graph)
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    ids = [joint_names.index(n) for n in chain.names]
    assert [joint_names[i] for i in env.action_manager.get_term(arm_term)._joint_ids] == chain.names
    assert env.action_manager.total_action_dim == 8 and np.isclose(env.step_dt,1/15)
    tool_id = body_names.index(graph['raw_tool_body'].split('/')[-1])
    tip_ids = [body_names.index(n) for n in tip_names]
    home_q = arr(robot.data.default_joint_pos)[0,ids].astype(float)
    root_initial = pose(arr(robot.data.root_state_w)[0])
    home_tool = chain.fk(home_q)
    command = home_q.copy()
    soft_limits = arr(robot.data.soft_joint_pos_limits)[0]
    stage = get_current_stage()
    geometry = {}
    for name in tip_names:
        matches = [p for p,v in graph['bodies'].items() if v['name'] == name]
        assert len(matches) == 1, matches
        runtime_path = '/World/envs/env_0/robot/' + matches[0].split('/',2)[2]
        geometry[name] = colliders(stage,runtime_path)
    (output/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    plan = {}
    seed = home_q.copy()
    for phase,_ in PHASES:
        target = home_tool.copy()
        if phase in ('x','y','z'):
            target['xyz'.index(phase),3] += .02
        if phase == 'rotate':
            target[:3,:3] = axis_rotation(np.array([0.,0.,1.]),.04)[:3,:3] @ target[:3,:3]
        result = solve(chain,target,seed,limits)
        plan[phase] = {'target_raw_tool_root':target.tolist(),'ik':result,'close':int(phase=='close')}
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        assert result['success'], (phase,result['reason'])
        seed = np.asarray(result['q'])
        assert np.max(np.abs(seed-home_q)) <= .3, 'Small-pose solution wandered from home'
    provenance = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ('check_pose_tracking.py','local_ik.py','rigid_chain.py')}
    inventory = {'robot':args.robot,'environment':names[0],'seed':0,'phases':PHASES,'steps':300,
        'raw_tool_body':graph['raw_tool_body'],'body_names':body_names,'joint_names':joint_names,'arm_indices':ids,
        'default_arm':home_q.tolist(),'root_initial':root_initial.tolist(),'source_sha256':provenance,
        'graph_sha256':hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        'camera_cfg':{k:env.scene[k].cfg.to_dict() for k in ('wrist_cam','over_shoulder_left_camera')},
        'thresholds':{'endpoint_position_m':.002,'endpoint_rotation_rad':.01,'endpoint_joint_rad':.01,
                      'gripper_endpoint_rad':.01,'command_max_increment_rad':.04,'measured_joint_margin_rad':.03},
        'scope':'Raw-tool tracking and native geometry/camera inventory only; no collision-free path, task, canonical-frame or policy claim.'}
    (output/'inventory.json').write_text(json.dumps(inventory,indent=2,default=str)+'\n')
    video = VideoWriter(str(output/'three_views.mp4'),15)
    checks = {}; snapshots = {}; wrist_relative_initial = None
    for phase,count in PHASES:
        goal = np.asarray(plan[phase]['ik']['q'])
        target = np.asarray(plan[phase]['target_raw_tool_root'])
        close = plan[phase]['close']
        for _ in range(count):
            pre_q = arr(robot.data.joint_pos)[0]
            previous_command = command.copy()
            command += np.clip(goal-command,-.04,.04)
            assert np.max(np.abs(command-previous_command)) <= .04000001
            assert np.isfinite(command).all() and (command>=limits[:,0]).all() and (command<=limits[:,1]).all()
            action = torch.as_tensor(np.r_[command,close],dtype=torch.float32,device=env.device)[None]
            obs,_,terminated,truncated,_ = env.step(action)
            assert not bool(terminated.any()) and not bool(truncated.any()), 'Unexpected episode reset'
            assert int(env.episode_length_buf[0]) == len(rows)+1
            q = arr(robot.data.joint_pos)[0]
            qd = arr(robot.data.joint_vel)[0]
            bodies = arr(robot.data.body_state_w)[0]
            root = pose(arr(robot.data.root_state_w)[0])
            tool = rigid_inverse(root) @ pose(bodies[tool_id])
            error = pose_error(target,tool)
            root_error = pose_error(root_initial,root)
            assert all(np.isfinite(x).all() for x in (q,qd,bodies,error,root_error))
            assert np.linalg.norm(root_error[:3])<1e-4 and np.linalg.norm(root_error[3:])<1e-4
            assert (q>=soft_limits[:,0]-.03).all() and (q<=soft_limits[:,1]+.03).all()
            fk_error = pose_error(chain.fk(q[ids]),tool)
            assert np.linalg.norm(fk_error[:3])<1e-4 and np.linalg.norm(fk_error[3:])<1e-3
            cameras = {}
            for name in ('wrist_cam','over_shoulder_left_camera'):
                data = env.scene[name].data
                camera = pose_matrix(arr(data.pos_w)[0],arr(data.quat_w_opengl)[0])
                k = arr(data.intrinsic_matrices)[0]
                assert np.isfinite(k).all() and k[0,0]>0 and k[1,1]>0
                cameras[name] = (camera,k)
            wrist_relative = rigid_inverse(root@tool) @ cameras['wrist_cam'][0]
            if wrist_relative_initial is None:
                wrist_relative_initial = wrist_relative.copy()
            camera_mount_error = pose_error(wrist_relative_initial,wrist_relative)
            rows.append({'step':len(rows)+1,'phase':phase,'action':arr(action)[0],
                'pre_q':pre_q,'q':q,'qd':qd,'root':root,'tool_root':tool,'target_root':target,
                'pose_error':error,'fk_error':fk_error,'root_error':root_error,
                'tip_world':np.stack([pose(bodies[i]) for i in tip_ids]),
                'camera_mount_error':camera_mount_error,
                'wrist_world':cameras['wrist_cam'][0],'wrist_k':cameras['wrist_cam'][1],
                'exterior_world':cameras['over_shoulder_left_camera'][0],'exterior_k':cameras['over_shoulder_left_camera'][1]})
            images = unpack_image_obs(obs)
            views = [images['over_shoulder_left_camera'],images['wrist_cam'],unpack_viewport_cams(obs)['combined_image']]
            panels=[]
            for label,image in zip(('EXTERIOR','WRIST','VIEWPORT'),views):
                assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3 and image.std()>1
                frame=cv2.resize(image,(640,360))
                cv2.putText(frame,f'{args.robot} {phase} {len(rows)} {label}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                panels.append(frame)
            video.write(cv2.hconcat(panels))
            assert np.linalg.norm(camera_mount_error[:3])<1e-4 and np.linalg.norm(camera_mount_error[3:])<1e-3, 'Wrist pose stale or mount inconsistent'
        check={'position_m':float(np.linalg.norm(error[:3])),'rotation_rad':float(np.linalg.norm(error[3:])),
               'joint_rad':float(np.max(np.abs(q[ids]-goal))),
               'gripper_rad':float(max(abs(q[joint_names.index(n)]-(v if close else 0.)) for n,v in grip_targets.items()))}
        checks[phase]=check
        (output/'endpoint_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        # Capture measured transforms for reconstructing enabled pad geometry in the raw-tool frame.
        raw_world=pose(bodies[tool_id]); inv_raw=rigid_inverse(raw_world)
        snapshots[phase]={'step':len(rows),'raw_tool_world':raw_world.tolist(),
            'tip_to_raw':{name:(inv_raw@pose(bodies[i])).tolist() for name,i in zip(tip_names,tip_ids)},
            'wrist_to_raw_opengl':(inv_raw@cameras['wrist_cam'][0]).tolist(),
            'wrist_k':cameras['wrist_cam'][1].tolist(),'exterior_world_opengl':cameras['over_shoulder_left_camera'][0].tolist(),
            'exterior_k':cameras['over_shoulder_left_camera'][1].tolist()}
        (output/'snapshots.json').write_text(json.dumps(snapshots,indent=2)+'\n')
        assert check['position_m']<=.002 and check['rotation_rad']<=.01 and check['joint_rad']<=.01, (phase,check)
        assert check['gripper_rad']<=.01, (phase,check)
    wrist_positions = np.stack([r['wrist_world'][:3,3] for r in rows])
    assert np.max(np.linalg.norm(wrist_positions-wrist_positions[0],axis=1))>.015, 'Wrist camera did not move with hand'
    complete=True
    (output/'checks.json').write_text(json.dumps({'complete':True,'steps':len(rows),'endpoints':checks,'scope':inventory['scope']},indent=2)+'\n')
    print('RAW_TOOL_TRACKING_COMPLETE',args.robot,len(rows),flush=True)
except BaseException:
    error=traceback.format_exc();(output/'failure.txt').write_text(error);print(error,flush=True)
    raise
finally:
    if rows:
        arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(output/'tracking.npz',complete=np.asarray(complete),**arrays)
    if video is not None:video.release()
    if env is not None:env.close()
    app.close()
