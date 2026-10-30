"""Distal-tip positive/negative controls with full substep localization; no policy."""
import argparse
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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert args.case.startswith('step6_B_contacts_') and all(c.isalnum() or c == '_' for c in args.case)
output = ROOT / 'runs' / args.case
output.mkdir(parents=True, exist_ok=False)
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from robolab.constants import set_output_dir
from robolab.core.environments.config import parse_env_cfg
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env
from robolab.core.observations.observation_utils import unpack_viewport_cams
from robolab.core.sensors.contact_sensor_utils import create_contact_sensors
from robolab.core.utils.video_utils import VideoWriter
from robolab.registrations.kinova.auto_env_registrations_jointpos import auto_register_kinova_envs
from robolab.robots.kinova_gen3 import ARM_JOINT_NAMES, _to_torch

TIP_NAMES = {side:f'robotiq_85_{side}_finger_tip_link' for side in ('left','right')}
RADIUS = .012
VERTEX_OVERLAP = .002
PHASES = [('away',20),('left',30),('away',20),('right',30),('away',20)]
ROBOT_BODIES = ['base_link','shoulder_link','half_arm_1_link','half_arm_2_link','forearm_link',
    'spherical_wrist_1_link','spherical_wrist_2_link','bracelet_link','bracelet_collision_box',
    'end_effector_link','robotiq_85_base_link','robotiq_85_left_inner_knuckle_link',
    'robotiq_85_left_knuckle_link','robotiq_85_right_inner_knuckle_link','robotiq_85_right_knuckle_link',
    'robotiq_85_left_finger_link','robotiq_85_right_finger_link','robotiq_85_left_finger_tip_link',
    'robotiq_85_right_finger_tip_link']


def arr(value):
    return _to_torch(value).detach().cpu().numpy().copy()


def rotation(q):
    q = np.asarray(q, dtype=float)
    assert np.isfinite(q).all() and abs(np.linalg.norm(q)-1) < 1e-4
    w,x,y,z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def collider_bounds(stage, body_path):
    body = stage.GetPrimAtPath(body_path)
    assert body.IsValid() and body.IsActive(), body_path
    # Native collision meshes have inherited purpose=guide, verified by static USD inspection.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render','proxy','guide'])
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    entries = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            bounds = cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
            lo,hi = np.array(bounds.GetMin()),np.array(bounds.GetMax())
            if np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(hi>lo):
                assert prim.IsA(UsdGeom.Mesh), str(prim.GetPath())
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                assert points.ndim == 2 and points.shape[1] == 3 and len(points) > 0 and np.isfinite(points).all()
                # USD/Gf uses row vectors: mesh->world followed by world->body.
                mesh_to_body = xcache.GetLocalToWorldTransform(prim) * xcache.GetLocalToWorldTransform(body).GetInverse()
                body_points = np.array([mesh_to_body.Transform(Gf.Vec3d(*p)) for p in points])
                homogeneous = np.column_stack((points,np.ones(len(points)))) @ np.array(mesh_to_body)
                assert np.allclose(homogeneous[:,3],1.)
                assert np.allclose(body_points,homogeneous[:,:3],atol=1e-10,rtol=1e-10)
                assert np.all(body_points >= lo-1e-6) and np.all(body_points <= hi+1e-6)
                entries.append({'path':str(prim.GetPath()),'purpose':str(UsdGeom.Imageable(prim).ComputePurpose()),
                    'collision_enabled':True,'min':lo.tolist(),'max':hi.tolist(),
                    'mesh_to_body_row_matrix':np.array(mesh_to_body).tolist(),
                    'mesh_vertices':points.tolist(),'body_vertices':body_points.tolist()})
    assert entries, body_path
    lo = np.min([v['min'] for v in entries], axis=0)
    hi = np.max([v['max'] for v in entries], axis=0)
    return {'colliders':entries,'min':lo.tolist(),'max':hi.tolist()}


def read_sensor(env, name, bodies, filters):
    sensor = env.scene[name]
    matrix = sensor.data.force_matrix_w
    assert matrix is not None and tuple(matrix.shape) == (1,1,len(filters),3), (name, None if matrix is None else matrix.shape)
    assert list(sensor.body_names) == bodies, (name,sensor.body_names)
    assert list(sensor.cfg.filter_prim_paths_expr) == filters
    force = arr(matrix)[0,0]
    assert np.isfinite(force).all(), name
    return force


def read_net(env, name):
    value=arr(env.scene[name].data.net_forces_w)
    assert value.shape == (1,1,3) and np.isfinite(value).all(), (name,value.shape)
    return value[0,0]


def main():
    set_output_dir(str(output))
    auto_register_kinova_envs(task='BananaInBowlTask')
    names = get_envs(task='BananaInBowlTask')
    assert len(names)==1, names
    cfg = parse_env_cfg(names[0], device=args.device, seed=0, num_envs=1, use_fabric=True)
    # Replace native multi-body sensors in this external diagnostic config only.
    for name in list(vars(cfg.scene)):
        if name.startswith('gripper__'):
            setattr(cfg.scene,name,None)
    cfg.contact_gripper = {s:f'{{ENV_REGEX_NS}}/robot/{n}' for s,n in TIP_NAMES.items()}
    cfg.contact_gripper['gripper'] = ['left','right']
    create_contact_sensors(cfg)
    cfg.scene.contact_probe = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/contact_probe',
        spawn=sim_utils.SphereCfg(radius=RADIUS, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
            mass_props=sim_utils.MassPropertiesCfg(mass=.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.,1.,0.))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.,2.,2.)))
    for side,name in TIP_NAMES.items():
        setattr(cfg.scene,side+'__probe',ContactSensorCfg(prim_path=f'{{ENV_REGEX_NS}}/robot/{name}',
            filter_prim_paths_expr=['{ENV_REGEX_NS}/contact_probe'],update_period=0.))
    cfg.scene.probe__tips = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/contact_probe',
        filter_prim_paths_expr=[f'{{ENV_REGEX_NS}}/robot/{name}' for name in TIP_NAMES.values()], update_period=0.)
    cfg.scene.probe__all_robot = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/contact_probe',
        filter_prim_paths_expr=[f'{{ENV_REGEX_NS}}/robot/{name}' for name in ROBOT_BODIES],update_period=0.)
    env,cfg = create_env(cfg)
    rows=[]; substeps=[]; complete=False
    original_scene_update=env.scene.update
    sample_context={'active':False}
    video = VideoWriter(str(output/'contact_controls.mp4'),15)
    try:
        obs,_ = env.reset()
        robot=env.scene['robot']; probe=env.scene['contact_probe']; stage=get_current_stage()
        assert robot.data.body_names == ROBOT_BODIES, robot.data.body_names
        assert cfg.decimation == 8 and np.isclose(cfg.sim.dt,1/120)
        arm_ids=[robot.data.joint_names.index(n) for n in ARM_JOINT_NAMES]
        assert [robot.data.joint_names[i] for i in env.action_manager.get_term('arm')._joint_ids] == ARM_JOINT_NAMES
        assert env.action_manager.total_action_dim==8 and np.isclose(env.step_dt,1/15)
        tip_ids={s:robot.data.body_names.index(n) for s,n in TIP_NAMES.items()}
        geometry={s:collider_bounds(stage,f'/World/envs/env_0/robot/{n}') for s,n in TIP_NAMES.items()}
        local_centers={}
        initial_bodies=arr(robot.data.body_state_w)[0]
        for side in TIP_NAMES:
            b=initial_bodies[tip_ids[side]]; other=initial_bodies[tip_ids['right' if side=='left' else 'left']]
            lo=np.array(geometry[side]['min']); hi=np.array(geometry[side]['max'])
            # Static mesh inspection places distal tips along +Z, clear of the proximal finger link.
            # v4 outer-X vertex contacted that neighboring link; localization_v1 proved its identity.
            axis=2; sign=1.
            assert np.argmax(hi-lo)==axis and hi[axis]>.05
            outward=np.zeros(3); outward[axis]=sign
            center=(lo+hi)/2
            prior_face=center.copy(); prior_face[axis]=hi[axis] if sign>0 else lo[axis]
            candidates=[(ci,vi,np.asarray(v)) for ci,c in enumerate(geometry[side]['colliders']) for vi,v in enumerate(c['body_vertices'])]
            extreme=max(sign*v[axis] for _,_,v in candidates)
            candidates=[item for item in candidates if np.isclose(sign*item[2][axis],extreme,atol=1e-9,rtol=0)]
            ci,vi,vertex=min(candidates,key=lambda item:(float(np.linalg.norm((item[2]-center)[np.arange(3)!=axis])),item[0],item[1]))
            point=vertex+outward*(RADIUS-VERTEX_OVERLAP)
            assert np.isclose(np.linalg.norm(point-vertex),RADIUS-VERTEX_OVERLAP,atol=1e-10,rtol=0)
            local_centers[side]=point
            geometry[side].update(probe_local_center=point.tolist(),face_axis=axis,face_sign=sign,
                selected_collider=geometry[side]['colliders'][ci]['path'],selected_vertex_index=vi,
                selected_body_vertex=vertex.tolist(),sphere_center_to_vertex_m=float(np.linalg.norm(point-vertex)),
                prior_AABB_face_midpoint=prior_face.tolist())
        legacy=stage.GetPrimAtPath('/World/envs/env_0/scene/franka_table')
        fixture=stage.GetPrimAtPath('/World/envs/env_0/table_fixture')
        assert fixture.IsValid() and fixture.IsActive()
        assert not legacy.IsValid() or not legacy.IsActive()
        sensor_inventory={}
        for name in ['left__probe','right__probe','probe__tips','probe__all_robot','left__banana','right__banana']:
            sensor=env.scene[name]
            sensor_inventory[name]={'body_names':list(sensor.body_names),'body_path':sensor.cfg.prim_path,
                'filters':list(sensor.cfg.filter_prim_paths_expr),'shape':list(sensor.data.force_matrix_w.shape)}
        inventory={'scope':'Diagnostic sphere contacts only; no grasp/policy/task success. Native banana single-body sensor identities verified separately.',
            'native_registration':names[0],'contact_gripper':cfg.contact_gripper,'geometry':geometry,
            'sphere_radius_m':RADIUS,'nominal_vertex_overlap_m':VERTEX_OVERLAP,
            'geometry_limit':'Sphere overlaps an actual enabled collision-mesh distal +Z vertex. This placement hypothesis requires positive and negative controls.',
            'sampling':'Every physics step (1/120s), eight samples per unchanged control step. Diagnostic probe follows displaced finger; not task motion.',
            'robot_body_filter_order':ROBOT_BODIES,'phases':PHASES,'settling_excluded_steps_per_phase':5,'positive_threshold_N':.1,
            'negative_threshold_N':.05,'opposite_pair_atol_N':.05,'opposite_pair_rtol':.001,
            'sensor_inventory':sensor_inventory,'fixture_active':True,'legacy_fixture_active':False,
            'force_convention':'tip->probe sensors report force on tip; probe->tips reports force on probe. Negate tip->banana vectors for force on banana.',
            'task_objects_manipulated':False}
        (output/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        home=_to_torch(robot.data.default_joint_pos)[0,arm_ids].clone()
        action=torch.cat((home,torch.zeros(1,device=env.device))).unsqueeze(0)
        probe_filters=[f'/World/envs/env_.*/robot/{name}' for name in TIP_NAMES.values()]
        all_robot_filters=[f'/World/envs/env_.*/robot/{name}' for name in ROBOT_BODIES]
        def sampled_scene_update(dt):
            original_scene_update(dt)
            if not sample_context['active']: return
            assert np.isclose(dt,1/120)
            tip_pair=np.stack([read_sensor(env,s+'__probe',[TIP_NAMES[s]],['/World/envs/env_.*/contact_probe'])[0] for s in TIP_NAMES])
            reciprocal=read_sensor(env,'probe__tips',['contact_probe'],probe_filters)
            all_pair=read_sensor(env,'probe__all_robot',['contact_probe'],all_robot_filters)
            tip_net=np.stack([read_net(env,s+'__probe') for s in TIP_NAMES])
            probe_net=read_net(env,'probe__tips')
            assert np.allclose(probe_net,read_net(env,'probe__all_robot'),atol=.05,rtol=.001)
            states=arr(robot.data.body_state_w)[0]
            pose_now=arr(probe.data.root_state_w)[0,:7]
            target_side=sample_context['side']
            vertex_now=np.zeros(3); vertex_distance=0.
            if target_side!='away':
                b=states[tip_ids[target_side]]
                vertex_now=b[:3]+rotation(b[3:7])@np.asarray(geometry[target_side]['selected_body_vertex'])
                vertex_distance=np.linalg.norm(pose_now[:3]-vertex_now)
            assert np.allclose(tip_pair,-reciprocal,atol=.05,rtol=.001)
            for ti,side in enumerate(TIP_NAMES):
                bi=ROBOT_BODIES.index(TIP_NAMES[side])
                assert np.allclose(reciprocal[ti],all_pair[bi],atol=.05,rtol=.001)
            q=arr(robot.data.joint_pos)[0]
            assert np.isfinite(states).all() and np.isfinite(pose_now).all() and np.isfinite(q).all()
            substeps.append({'physics_sample':len(substeps)+1,'control_step':sample_context['control_step'],
                'phase':sample_context['phase'],'dwell':sample_context['dwell'],
                'probe_command_pose':sample_context['command'].copy(),'probe_measured_pose':pose_now,
                'tip_pair_forces':tip_pair,'probe_tip_forces':reciprocal,'probe_robot_pair_forces':all_pair,
                'tip_net_forces':tip_net,'probe_net_force':probe_net,
                'unassigned_probe_force':probe_net-all_pair.sum(axis=0),
                'tip_world_poses':np.stack([states[tip_ids[s],:7] for s in TIP_NAMES]),
                'target_vertex_world':vertex_now,'target_vertex_distance_m':vertex_distance,
                'joint_positions':q})
        env.scene.update=sampled_scene_update
        checks=[]
        for phase_index,(side,count) in enumerate(PHASES):
            start=len(rows)
            for dwell in range(count):
                bodies=arr(robot.data.body_state_w)[0]
                vertex_world=np.zeros(3); command_vertex_distance=0.
                if side=='away': position=np.array([2.,2.,2.])
                else:
                    b=bodies[tip_ids[side]]
                    position=b[:3]+rotation(b[3:7])@local_centers[side]
                    vertex_world=b[:3]+rotation(b[3:7])@np.asarray(geometry[side]['selected_body_vertex'])
                    command_vertex_distance=float(np.linalg.norm(position-vertex_world))
                    assert np.isclose(command_vertex_distance,RADIUS-VERTEX_OVERLAP,atol=1e-9,rtol=0)
                pose=torch.tensor([[*position,1.,0.,0.,0.]],device=env.device,dtype=torch.float32)
                probe.write_root_pose_to_sim(pose)
                probe.write_root_velocity_to_sim(torch.zeros((1,6),device=env.device))
                substep_start=len(substeps)
                sample_context.update(active=True,control_step=len(rows)+1,phase=phase_index,dwell=dwell,side=side,command=pose.cpu().numpy()[0])
                obs,_,terminated,truncated,_=env.step(action)
                sample_context['active']=False
                assert len(substeps)-substep_start == 8, len(substeps)-substep_start
                assert not bool(terminated.any()) and not bool(truncated.any())
                assert int(env.episode_length_buf[0]) == len(rows)+1
                q=arr(robot.data.joint_pos)[0]; assert np.isfinite(q).all()
                tip_forces=np.stack([read_sensor(env,s+'__probe',[TIP_NAMES[s]],['/World/envs/env_.*/contact_probe'])[0] for s in TIP_NAMES])
                probe_forces=read_sensor(env,'probe__tips',['contact_probe'],probe_filters)
                assert np.allclose(tip_forces,-probe_forces,atol=.05,rtol=.001), (tip_forces,probe_forces)
                banana_forces=np.stack([read_sensor(env,s+'__banana',[TIP_NAMES[s]],['/World/envs/env_.*/scene/banana'])[0] for s in TIP_NAMES])
                measured_pose=arr(probe.data.root_state_w)[0,:7]
                vertex_after=np.zeros(3); measured_vertex_distance=0.
                if side!='away':
                    b_after=arr(robot.data.body_state_w)[0,tip_ids[side]]
                    vertex_after=b_after[:3]+rotation(b_after[3:7])@np.asarray(geometry[side]['selected_body_vertex'])
                    measured_vertex_distance=float(np.linalg.norm(measured_pose[:3]-vertex_after))
                rows.append({'step':len(rows)+1,'phase':phase_index,'dwell':dwell,'probe_command_pose':pose.cpu().numpy()[0],
                    'probe_measured_pose':measured_pose,'tip_forces':tip_forces,'probe_forces':probe_forces,
                    'tip_net_forces':np.stack([read_net(env,s+'__probe') for s in TIP_NAMES]),
                    'probe_net_force':read_net(env,'probe__tips'),'target_vertex_world_before_step':vertex_world,
                    'command_vertex_distance_m':command_vertex_distance,
                    'target_vertex_world_after_step':vertex_after,'measured_vertex_distance_m':measured_vertex_distance,
                    'banana_forces':banana_forces,'joint_positions':q,
                    'tip_world_poses':np.stack([arr(robot.data.body_state_w)[0,tip_ids[s],:7] for s in TIP_NAMES])})
                frame=unpack_viewport_cams(obs)['combined_image'].copy()
                cv2.putText(frame,f'CONTACT CONTROL: {side} | {dwell+1}/{count}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),1,cv2.LINE_AA)
                video.write(frame)
            forces=np.stack([r['tip_forces'] for r in rows[start+5:]])
            norms=np.linalg.norm(forces,axis=-1)
            check={'phase':phase_index,'target':side,'max_tip_force_N':norms.max(axis=0).tolist()}
            if side=='away': assert (norms<.05).all(),check
            else:
                intended=list(TIP_NAMES).index(side); other=1-intended
                check['positive_samples']=int((norms[:,intended]>.1).sum())
                assert check['positive_samples']>=5 and (norms[:,other]<.05).all(),check
            settled=[r for r in substeps if r['phase']==phase_index and r['dwell']>=5]
            all_pair=np.stack([r['probe_robot_pair_forces'] for r in settled])
            all_norms=np.linalg.norm(all_pair,axis=-1)
            residual=np.linalg.norm(np.stack([r['unassigned_probe_force'] for r in settled]),axis=-1)
            assert (residual<.05).all(), ('Unassigned collision force',residual.max())
            if side=='away': assert (all_norms<.05).all()
            else:
                target_body=ROBOT_BODIES.index(TIP_NAMES[side])
                assert (all_norms[:,np.arange(len(ROBOT_BODIES))!=target_body]<.05).all(), 'Other robot link contact'
            check['substep_pair_max_N']=all_norms.max(axis=0).tolist()
            check['substep_max_unassigned_N']=float(residual.max())
            checks.append(check)
            (output/'partial_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        assert len(substeps)==960 and len(rows)==120
        for key in substeps[0]: assert np.isfinite(np.stack([r[key] for r in substeps])).all(),key
        localization={name:float(np.linalg.norm(np.stack([r['probe_robot_pair_forces'][i] for r in substeps]),axis=-1).max()) for i,name in enumerate(ROBOT_BODIES)}
        (output/'localization.json').write_text(json.dumps({'body_filter_order':ROBOT_BODIES,'max_pair_force_N':localization,
            'max_unassigned_force_N':float(np.linalg.norm(np.stack([r['unassigned_probe_force'] for r in substeps]),axis=-1).max()),
            'scope':'Distal sphere controls only; filtering proof is separate from banana grasp or task success.'},indent=2)+'\n')
        complete=True
        (output/'checks.json').write_text(json.dumps({'complete':True,'steps':len(rows),'checks':checks,
            'physics_samples':len(substeps),'scope':'Both tip positive/negative sphere controls verified at declared thresholds; no banana grasp/policy/task success.'},indent=2)+'\n')
    except BaseException:
        # SimulationApp.close may exit the process; persist the failure before cleanup.
        failure=traceback.format_exc()
        (output/'failure.txt').write_text(failure)
        print(failure,flush=True)
        raise
    finally:
        sample_context['active']=False
        env.scene.update=original_scene_update
        if substeps: np.savez_compressed(output/'physics_contacts.npz',complete=np.asarray(complete),**{k:np.stack([r[k] for r in substeps]) for k in substeps[0]})
        if rows: np.savez_compressed(output/'contacts.npz',complete=np.asarray(complete),**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        video.release(); env.close(); app.close()


if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc(); app.close(); sys.exit(1)
