"""Exploratory development round on the painted arm (after the six-episode live comparison, run_painted_bridge.py).
Identical to run_painted_bridge.py with --arm painted, except ONE declared fixed adapter change per --variant:
  lift: while P0 has commanded close for more than LIFT_DELAY controls, B's commanded tool target is raised LIFT_M in
        world +Z (ramped LIFT_M/LIFT_RAMP per control, up and down), but only for a close that began with the banana
        neither touching the bowl nor inside the bowl predicate (previous control's signals), so the adapter never lifts
        a banana already resting on the bowl. Observation, painting and P0's input pipeline are unchanged.
  grip: B's six-joint Robotiq gripper actuator stiffness and damping are multiplied by GRIP_GAIN_SCALE (effort limit
        unchanged). Everything else unchanged.
Scored online by the same B-ported v4 gate; stops on gate success. Exploratory: not part of the frozen comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path.home()/'robot-transfer/bowl'
REPO = ROOT/'RoboLab'
sys.path.insert(0,str(REPO)); os.chdir(REPO)
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--case',required=True)
parser.add_argument('--scene',required=True,choices=('layout','base'))
parser.add_argument('--arm',required=True,choices=('painted',))
parser.add_argument('--variant',required=True,choices=('lift',))
parser.add_argument('--max-ticks',type=int,default=None)  # smoke only
parser.add_argument('--layout-seed',type=int,default=None)  # reserved step 8 layout (scene layout only)
parser.add_argument('--policy',required=True,choices=('P0','P1'))  # label only; the supervisor proves identity by pin
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
POLICY_SEED=int(os.environ['POLICY_SEED']);assert POLICY_SEED==0
SMOKE=args.case=='step8_B_smoke_l2000_s0_v1'
if SMOKE:assert args.scene=='layout' and args.policy=='P0' and args.layout_seed==2000 and args.max_ticks==300  # declared non-scoring check of the layout branch and A->B pose hand-off on a training layout, never a test layout
else:assert args.scene=='layout' and args.layout_seed in range(4000,4012) and args.max_ticks is None and args.case in (f'step8_B_{args.policy}_l{args.layout_seed}_s{POLICY_SEED}_v1',f'step8_B_P0canary_l{args.layout_seed}_s{POLICY_SEED}_v1' if args.policy=='P0' else None)  # optional after-phase drift canary
assert all(c.isalnum() or c=='_' for c in args.case)
output=ROOT/'runs'/args.case
output.mkdir(parents=True,exist_ok=False)
args.enable_cameras=True
app=AppLauncher(args).app
import signal
signal.signal(signal.SIGTERM,lambda *a:sys.exit(143))  # a supervisor timeout still runs except/finally and keeps the evidence
rows=[]; gate_trace=[]; env=video=client=capture=trace=None; complete=False
try:
    import cv2
    import h5py
    import numpy as np
    import torch
    import zmq
    from rigid_chain import RigidChain,pose_matrix,rigid_inverse
    from local_ik import solve,physical_limits,pose_error
    from robolab.constants import set_output_dir
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.observations.observation_utils import unpack_image_obs,unpack_viewport_cams
    from robolab.core.utils.video_utils import VideoWriter
    from robolab.core.sensors.contact_sensor_utils import create_contact_sensors,get_contact_sensor_with_order
    from robolab.core.world.world_state import get_world
    from robolab.core.task.conditionals import object_in_container
    from robolab.core.task.instruction import primary_instruction
    from robolab.registrations.kinova.auto_env_registrations_jointpos import auto_register_kinova_envs
    from robolab.robots.kinova_gen3 import _to_torch,GRIPPER_JOINT_NAME
    from policies.gr00t.client import GR00TDroidJointposClient,DROID_EEF_ROTATION_CORRECT,_get_action_value,_as_action_chunk
    sys.path.insert(0,str(ROOT/'experiments/bowl_step5'))
    from release_gate_v4 import ReleaseGateV4
    from rim_support import support_signals
    from paint_A import add_painted_A,Painter,EXTERIOR
    LOGICAL=750;REPEAT=12;TICKS=LOGICAL*REPEAT  # A's 50 s logical budget (750 actions), each held 12 controls
    RUN_TICKS=args.max_ticks if SMOKE else TICKS
    RELEASE_TIMEOUT_B_S=.5+24/15  # A's 0.5 s + B's measured full gripper travel (24 controls at 1/24 per tick, gripper_timing_v1)
    VIDEO_EVERY=3
    LIFT_M=.015;LIFT_DELAY=24;LIFT_RAMP=24  # lift: 1.5 cm, after one full gripper travel (24 controls), ramped over another 24
    GRIP_GAIN_SCALE=5.  # grip: stiffness 100 -> 500, damping 5 -> 25

    def arr(x):return _to_torch(x).detach().cpu().numpy().copy()
    def pose(x):return pose_matrix(x[:3],x[3:7])
    def save(name,value): (output/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def log_solve(step,direction,result,target,seed):
        result={**result,'target':np.asarray(target).tolist(),'seed':np.asarray(seed).tolist()}
        trace.write(json.dumps({'step':step,'direction':direction,**result})+'\n');trace.flush()
        if not result['success']:save('failed_solve.json',{'step':step,'direction':direction,**result})
        assert result['success'], (step,direction,result['reason'])

    paths={'frames':ROOT/'runs/step6_fixed_frames_cpu_v1/frame_proposal.json','gate':ROOT/'experiments/bowl_step5/release_gate_v4.py','gate_v3':ROOT/'experiments/bowl_step5/release_gate_v3.py',
           'rim':ROOT/'experiments/bowl_step5/rim_support.py','stability':ROOT/'experiments/bowl_step5/stability.py',
           'initial_ik':ROOT/'runs/step6_fixed_frames_cpu_v1/ik_results.json',
           'A_inventory':ROOT/'runs/step6_A_tracking_v3/inventory.json',
           'graph':ROOT/'assets/report/evidence/2026-10-21/bowl-step6/kinematics_export_v1/joint_graphs.json'}
    frames=json.loads(paths['frames'].read_text()); initial_ik=json.loads(paths['initial_ik'].read_text())
    graphs=json.loads(paths['graph'].read_text()); A=RigidChain(graphs['A']); B=RigidChain(graphs['B'])
    Alim=physical_limits(graphs['A']); Blim=physical_limits(graphs['B'])
    CA,CB=(np.asarray(frames['raw_T_canonical'][r]) for r in ('A','B'))
    Aroot=np.asarray(frames['root_world']['A']); Acam=np.asarray(frames['A_raw_T_camera_opengl'])
    initialB=np.asarray(initial_ik[0]['forward']['q']);assert initial_ik[0]['name']=='matched_A_home' and initial_ik[0]['forward']['success']
    virtual_seed=np.asarray(json.loads(paths['A_inventory'].read_text())['default_arm'])
    set_output_dir(str(output))
    if args.scene=='layout':
        auto_register_kinova_envs(task_dirs=['randomize_initial_pose'],task='BananaInBowlUniformInitPose10cmTask')
        names=get_envs(task='BananaInBowlTask');assert names==['BananaInBowlUniformInitPose10cmTaskKinovaJointPosition'],names;layout_seed=args.layout_seed
        Aref=REPO/f"output/step8_A_{'smoke' if SMOKE else 'P0'}_l{layout_seed}_s{POLICY_SEED}_v1/BananaInBowlUniformInitPose10cmTask"  # both B arms read A/P0's poses for this layout
    else:
        auto_register_kinova_envs(task='BananaInBowlTask');names=get_envs(task='BananaInBowlTask');assert len(names)==1;layout_seed=0
        Aref=REPO/f'output/step5_controlled_base_s{POLICY_SEED}_v3/BananaInBowlTask'
    Ametric=json.loads((Aref/'strict-metric.json').read_text());assert Ametric['layout_seed']==layout_seed and Ametric['policy_seed']==POLICY_SEED
    cfg=parse_env_cfg(names[0],device=args.device,seed=layout_seed,num_envs=1,use_fabric=True)
    original_success=str(cfg.terminations.success);rawfunc=cfg.terminations.success.func;rawparams=dict(cfg.terminations.success.params)
    cfg.terminations.success=None  # replaced by the online B-ported gate below; every other native termination kept
    cfg.episode_length_s=602  # 9001 controls = 600.07 s
    for n in list(vars(cfg.scene)):
        if n.startswith('gripper__'):setattr(cfg.scene,n,None)
    tips={s:f'robotiq_85_{s}_finger_tip_link' for s in ('left','right')}
    cfg.contact_gripper={s:f'{{ENV_REGEX_NS}}/robot/{n}' for s,n in tips.items()}
    cfg.contact_gripper['gripper']=['left','right'];create_contact_sensors(cfg)
    cfg.scene.robot.init_state.joint_pos.update({f'joint_{i+1}':float(v) for i,v in enumerate(initialB)})
    gripper_actuator=cfg.scene.robot.actuators['gripper'];gripper_gains={'stiffness':gripper_actuator.stiffness,'damping':gripper_actuator.damping,'effort_limit':gripper_actuator.effort_limit}
    assert gripper_gains=={'stiffness':100.,'damping':5.,'effort_limit':50.},gripper_gains
    if args.variant=='grip':gripper_actuator.stiffness*=GRIP_GAIN_SCALE;gripper_actuator.damping*=GRIP_GAIN_SCALE
    for n in ('wrist_cam','over_shoulder_left_camera'):getattr(cfg.scene,n).update_latest_camera_pose=True
    add_painted_A(cfg)
    cam=frames['B_camera_cfg'];cfg.scene.wrist_cam.prim_path=cam['prim_path']
    cfg.scene.wrist_cam.offset.pos=tuple(cam['offset']['pos']);cfg.scene.wrist_cam.offset.rot=tuple(cam['offset']['rot'])
    cfg.scene.wrist_cam.offset.convention='opengl'
    for n in ('width','height'):setattr(cfg.scene.wrist_cam,n,cam[n])
    for n in ('focal_length','horizontal_aperture','vertical_aperture'):setattr(cfg.scene.wrist_cam.spawn,n,cam[n])
    env,cfg=create_env(cfg,instruction_type='default');obs,_=env.reset()
    # Same scene as A's reference trial: restore A's recorded initial object poses (direct state write, no physics step).
    for n in ('banana','bowl'):
        env.scene[n].write_root_state_to_sim(torch.as_tensor(np.asarray(Ametric['initial_poses_world'][n]),device=env.device,dtype=torch.float32)[None])
    env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.0);env.sim.render();obs=env.observation_manager.compute()
    for n in ('banana','bowl'):assert np.allclose(_to_torch(env.scene[n].data.root_state_w).cpu().numpy()[0,:7],Ametric['initial_poses_world'][n][:7],atol=1e-5,rtol=0),n
    painter=Painter(env);world=get_world(env);gate=ReleaseGateV4(release_timeout_s=RELEASE_TIMEOUT_B_S)
    def release_signals_B():
        # Port of run_controlled_episode_v4_registered.release_signals. Declared differences: closure is B's
        # robotiq_85_left_knuckle_joint / 0.8 (A: finger_joint / (pi/4)); the 'gripper' contact group is B's two Robotiq
        # fingertip pads, as in the bridge since contacts_v5.
        boxes={n:world.get_bbox(n)[0][0].detach().cpu().numpy() for n in ('banana','bowl','table')}
        # The bridge's 'gripper' label is an alias group of the two tip sensors; there is no gripper__banana sensor.
        sensors={n:get_contact_sensor_with_order(env.scene,'banana',n) for n in ('bowl','table')}
        forces={n:sensor.data.force_matrix_w[0].detach().cpu().numpy() for n,(sensor,_) in sensors.items()}
        forces['gripper']=np.concatenate([env.scene[f'{s}__banana'].data.force_matrix_w[0].detach().cpu().numpy() for s in ('left','right')])  # sign unused: |f|>0.1
        support=support_signals(boxes['banana'],boxes['bowl'],forces['bowl'],sensors['bowl'][1])
        closure=float(robot.data.joint_pos[0,grip_id].item())/.8
        finite=bool(all(np.isfinite(v).all() for v in list(boxes.values())+list(forces.values())) and np.isfinite(closure))
        contacts={n:bool(np.any(np.abs(f)>.1)) for n,f in forces.items()}
        clearance=float(boxes['banana'][:,2].min()-boxes['table'][:,2].max());height=float(boxes['banana'][:,2].mean()-boxes['bowl'][:,2].max())
        geometric=bool(object_in_container(env,object='banana',container='bowl',require_contact_with=False,require_gripper_detached=False,env_id=0)) if finite else False
        return {**support,'finite':finite,'contact':contacts['gripper'],'table_contact':contacts['table'],'bowl_contact':contacts['bowl'],'closure':closure,
                'clearance_m':clearance,'near_bowl':geometric and height<=.15,'geometric_bowl_predicate':geometric,'centroid_height_above_bowl_m':height}
    robot=env.scene['robot'];joints=list(robot.data.joint_names);bodies=list(robot.data.body_names)
    grip_id=joints.index(GRIPPER_JOINT_NAME)
    ids=[joints.index(n) for n in B.names];tool_id=bodies.index(graphs['B']['raw_tool_body'].split('/')[-1]);grip_id=joints.index(GRIPPER_JOINT_NAME)
    assert [joints[i] for i in env.action_manager.get_term('arm')._joint_ids]==B.names
    assert env.action_manager.total_action_dim==8 and np.isclose(env.step_dt,1/15)
    root0=pose(arr(robot.data.root_state_w)[0]);assert np.allclose(root0,frames['root_world']['B'],atol=1e-7,rtol=0)
    assert np.allclose(arr(robot.data.joint_pos)[0,ids],initialB,atol=1e-6,rtol=0)
    soft=arr(robot.data.soft_joint_pos_limits)[0];command=initialB.copy()
    _,instruction=primary_instruction(cfg.instruction)
    client=GR00TDroidJointposClient(remote_host='127.0.0.1',remote_port=15555,open_loop_horizon=8)
    client.client.socket.setsockopt(zmq.RCVTIMEO,60000);client.client.socket.setsockopt(zmq.SNDTIMEO,60000);client.client.socket.setsockopt(zmq.LINGER,0)
    video=VideoWriter(str(output/'views.mp4'),15//VIDEO_EVERY);last_painted=None;success_step=None
    capture=h5py.File(output/'requests.hdf5','x');trace=(output/'ik_trace.jsonl').open('x')
    (output/'responses').mkdir()
    inventory={'scope':'Step 8 final-test episode on a reserved layout (or the declared non-scoring smoke check): frozen B adapter v1 (bridge + exterior painting + 15 mm lift), POLICY_SEED 0, B initial poses from A/P0 on the same layout. Never used for tuning.','policy_label':args.policy,'layout_seed_argument':args.layout_seed,'frozen_record':'frozen_B_adapter_v1r2',
        'variant':args.variant,'variant_details':{'lift':{'lift_m':LIFT_M,'delay_controls':LIFT_DELAY,'ramp_controls':LIFT_RAMP,'frame':'world +Z on the commanded B tool target','trigger':'P0 gripper command >0.5 held for more than delay_controls consecutive controls','eligibility':'only a close that began with the banana not touching the bowl and outside the bowl predicate (previous control signals)','observation':'unchanged (B measured pose)'},
                                                 'grip':{'gain_scale':GRIP_GAIN_SCALE,'baseline_gains':gripper_gains,'applied':{'stiffness':gripper_actuator.stiffness,'damping':gripper_actuator.damping,'effort_limit':gripper_actuator.effort_limit}}}[args.variant],
        'arm':args.arm,'scene':args.scene,'A_reference':str(Aref),'environment':names[0],'layout_seed':layout_seed,'policy_seed':POLICY_SEED,'instruction':instruction,'max_steps':TICKS,'logical_actions':LOGICAL,'repeat_factor':REPEAT,'replan_every_physical_ticks':96,'replan_interval_sim_s':6.4,'initialization_steps':1,'total_simulator_steps':TICKS+1,'hz':15,'open_loop_horizon':8,
        'painting':{'module':'paint_A.py','computed_in_both_arms':True,'P0_receives':'painted exterior' if args.arm=='painted' else 'actual B exterior','wrist_painted':False,
                    'simulator_information_used':'B measured joints -> virtual-A IK (existing observation bridge); B knuckle angle; simulator renderer with B visuals hidden for the painted render','painter':painter.inventory},
        'scoring':{'gate':'release_gate_v4.ReleaseGateV4 (frozen Step 5 v4 logic), fed before/after every control tick, stop on success','release_timeout_s':RELEASE_TIMEOUT_B_S,
                   'declared_deviations_from_A':['closure = B knuckle/0.8','gripper contact = B Robotiq fingertip pads','release timeout 0.5 s + 1.6 s B full gripper travel'],
                   'upstream_predicate':original_success,'independent_adjudication':'separate; never pooled with the automatic score'},
        'video':f'every {VIDEO_EVERY}th control at {15//VIDEO_EVERY} fps: B exterior (actual) | P0 exterior (painted, latest request) | wrist',
        'initial_B':initialB.tolist(),'root_initial':root0.tolist(),'arm_indices':ids,'joint_names':joints,
        'camera_remount':cam,'command_max_increment_rad':.04,'measured_joint_margin_rad':.03,
        'gripper_observation':'clip measured robotiq_85_left_knuckle_joint / 0.8 to [0,1]; A observation corruption is disabled',
        'gripper_action':'P0 >0.5 closes, otherwise opens; native B six-joint binary target',
        'termination_change':{'removed_success':original_success,'replaced_by':'online B-ported v4 gate','native_timeout_replaced_s':602,'policy_duration_s':600,'total_simulator_duration_s':9001/15,
                              'other_native_terminations':'kept; if one fires it is recorded as an episode outcome (no success), not a crash'},
        'smoke':SMOKE,'run_ticks':RUN_TICKS,
        'action_mapping':'A predicted joint FK -> A canonical world -> B raw root -> local IK seeded from measured B; rate limit from previous command',
        'observation_mapping':'B measured canonical world -> virtual A local IK, continuous previous virtual seed; internally consistent virtual FK pose',
        'auxiliary_eef_action':'saved, never executed','setup_hashes':{str(p):digest(p) for p in paths.values()},
        'source_hashes':{n:digest(Path(__file__).with_name(n)) for n in ('run_painted_dev.py','paint_A.py','local_ik.py','rigid_chain.py')},
        'client_source_hash':digest(REPO/'policies/gr00t/client.py'),'fixed_frames':frames}
    try:  # the gains PhysX actually holds for B's six gripper joints, not just the cfg
        gripper_joint_ids=[joints.index(n) for n in gripper_actuator.joint_names_expr]
        inventory['gripper_sim_drive']={'joints':list(gripper_actuator.joint_names_expr),'stiffness':arr(robot.root_physx_view.get_dof_stiffnesses())[0,gripper_joint_ids].tolist(),
                                        'damping':arr(robot.root_physx_view.get_dof_dampings())[0,gripper_joint_ids].tolist()}
    except Exception as exc:inventory['gripper_sim_drive']={'unavailable':repr(exc)}
    save('inventory.json',inventory)
    def state():
        q=arr(robot.data.joint_pos)[0];qd=arr(robot.data.joint_vel)[0]
        root=pose(arr(robot.data.root_state_w)[0]);tool=rigid_inverse(root)@pose(arr(robot.data.body_state_w)[0,tool_id])
        assert all(np.isfinite(v).all() for v in (q,qd,root,tool))
        re=pose_error(root0,root);fe=pose_error(B.fk(q[ids]),tool)
        assert np.linalg.norm(re[:3])<1e-4 and np.linalg.norm(re[3:])<1e-4
        assert np.linalg.norm(fe[:3])<1e-4 and np.linalg.norm(fe[3:])<1e-3
        assert (q>=soft[:,0]-.03).all() and (q<=soft[:,1]+.03).all()
        return q,qd,root,tool
    def startup_snapshot(label,current_obs):
        q,qd,root,tool=state()
        data=env.scene['wrist_cam'].data
        camera=pose_matrix(arr(data.pos_w)[0],arr(data.quat_w_opengl)[0])
        expected=root@tool@CB@rigid_inverse(CA)@Acam
        error=pose_error(expected,camera)
        record={'q':q.tolist(),'qd':qd.tolist(),'root':root.tolist(),'tool_root':tool.tolist(),
                'camera_world':camera.tolist(),'expected_camera_world':expected.tolist(),
                'camera_error':error.tolist(),'K':arr(data.intrinsic_matrices)[0].tolist(),
                'objects':{n:arr(env.scene[n].data.root_state_w)[0].tolist() for n in ('banana','bowl')},
                'episode_step':int(env.episode_length_buf[0])}
        save('startup_'+label+'.json',record)
        imgs=unpack_image_obs(current_obs)
        imgs={n:imgs[n] for n in ('wrist_cam','over_shoulder_left_camera')}
        imgs['viewport']=unpack_viewport_cams(current_obs)['combined_image']
        for n,img in imgs.items():
            assert img.dtype==np.uint8 and img.ndim==3 and img.shape[2]==3
            assert cv2.imwrite(str(output/f'startup_{label}_{n}.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        return record
    startup_snapshot('before',obs)
    # Exactly one declared initialization action; never add adaptive warmup/retry steps.
    warm_action=torch.as_tensor(np.r_[initialB,0.],device=env.device,dtype=torch.float32)[None]
    obs,_,term,trunc,_=env.step(warm_action)
    assert not bool(term.any()) and not bool(trunc.any()) and int(env.episode_length_buf[0])==1
    startup=startup_snapshot('after',obs)
    error=np.asarray(startup['camera_error'])
    assert np.linalg.norm(error[:3])<1e-4 and np.linalg.norm(error[3:])<1e-3, 'Camera still mismatched after one initialization step'
    assert np.allclose(startup['K'],frames['expected_camera_K'],atol=1e-4,rtol=0)
    native_termination=None;lift=0.;closed_run=0;lift_ok=False;post=None
    for step in range(RUN_TICKS):
        logical_step=step//12
        repeat_index=step%12
        q,qd,root,tool=state()
        targetA=rigid_inverse(Aroot)@root@tool@CB@rigid_inverse(CA)
        inverse=solve(A,targetA,virtual_seed,Alim);log_solve(step,'observation_B_to_A',inverse,targetA,virtual_seed)
        virtual_seed=np.asarray(inverse['q']);virtual_tool=A.fk(virtual_seed)
        eef=np.r_[virtual_tool[:3,3],(virtual_tool[:3,:3]@DROID_EEF_ROTATION_CORRECT)[:2,:].reshape(-1)].astype(np.float32)
        images=unpack_image_obs(obs)
        closure=np.asarray([np.clip(q[grip_id]/.8,0,1)],dtype=np.float32)
        extracted={'external_image':images['over_shoulder_left_camera'],'wrist_image':images['wrist_cam'],
                   'joint_position':virtual_seed.astype(np.float32),'gripper_position':closure,'eef_9d':eef}
        fresh=step%96==0
        if fresh:
            painted,written=painter.paint(virtual_seed,float(closure[0]));last_painted=painted
            packed_actual=client._pack_request(extracted,instruction)
            packed_painted=client._pack_request({**extracted,'external_image':painted},instruction)
            packed=packed_painted if args.arm=='painted' else packed_actual
        camdata=env.scene['wrist_cam'].data
        actualcam=pose_matrix(arr(camdata.pos_w)[0],arr(camdata.quat_w_opengl)[0])
        ce=pose_error(root@tool@CB@rigid_inverse(CA)@Acam,actualcam)
        vce=pose_error(Aroot@virtual_tool@Acam,actualcam)
        if not (np.linalg.norm(ce[:3])<1e-4 and np.linalg.norm(ce[3:])<1e-3):
            save('failed_camera.json',{'policy_step':step,'actual':actualcam.tolist(),'physical_expected':(root@tool@CB@rigid_inverse(CA)@Acam).tolist(),'error':ce.tolist()})
        assert np.linalg.norm(ce[:3])<1e-4 and np.linalg.norm(ce[3:])<1e-3
        assert np.linalg.norm(vce[:3])<3e-4 and np.linalg.norm(vce[3:])<.002
        assert np.allclose(arr(camdata.intrinsic_matrices)[0],frames['expected_camera_K'],atol=1e-4,rtol=0)
        if fresh:
            group=capture.create_group(str(step));group.attrs['instruction']=instruction;group.attrs['P0_exterior']=args.arm
            for k,v in packed.items():
                if k.startswith(('video.','state.')):group.create_dataset(k,data=v,compression='lzf')
            group.create_dataset('actual.video.exterior_image_1_left',data=packed_actual['video.exterior_image_1_left'],compression='lzf')
            group.create_dataset('painted.video.exterior_image_1_left',data=packed_painted['video.exterior_image_1_left'],compression='lzf')
            group.create_dataset('painted_A_joint_position',data=written)
            capture.flush()
            response=client.client.get_action(packed)
            chunks={n:_as_action_chunk(_get_action_value(response[0],n),name=n) for n in ('joint_position','gripper_position','eef_9d')}
            np.savez_compressed(output/'responses'/f'{step:04d}.npz',**chunks)
            assert all(v.shape==(40,d) and np.isfinite(v).all() for (n,v),d in zip(chunks.items(),(7,1,9)))
            chunk=client._postprocess_chunk(client._unpack_response(response));assert chunk.shape==(40,8)
        predicted=chunk[logical_step%8];Aq=predicted[:7].astype(float)
        assert (Aq>=Alim[:,0]).all() and (Aq<=Alim[:,1]).all(), 'P0 A joint target outside physical limits'
        targetB=rigid_inverse(root)@Aroot@A.fk(Aq)@CA@rigid_inverse(CB)
        if args.variant=='lift':
            if predicted[7]>.5:
                if closed_run==0:lift_ok=not (post is not None and (post['bowl_contact'] or post['geometric_bowl_predicate']))  # decided at the close edge
                closed_run+=1
            else:closed_run=0
            lift+=float(np.clip((LIFT_M if closed_run>LIFT_DELAY and lift_ok else 0.)-lift,-LIFT_M/LIFT_RAMP,LIFT_M/LIFT_RAMP))
            targetB[:3,3]+=rigid_inverse(root)[:3,:3]@np.array([0.,0.,lift])
        forward=solve(B,targetB,q[ids],Blim);log_solve(step,'action_A_to_B',forward,targetB,q[ids])
        goal=np.asarray(forward['q']);previous=command.copy();command+=np.clip(goal-command,-.04,.04)
        assert np.max(np.abs(command-previous))<=.04000001
        assert np.isfinite(command).all() and (command>=Blim[:,0]).all() and (command<=Blim[:,1]).all()
        action=torch.as_tensor(np.r_[command,predicted[7]],device=env.device,dtype=torch.float32)[None]
        pre=release_signals_B();t_pre=int(env.episode_length_buf[0])*float(env.step_dt);gate.before(t_pre,float(predicted[7]),pre)
        obs,_,term,trunc,_=env.step(action)
        if bool(term.any()) or bool(trunc.any()):
            native_termination={'step':step+1,'terminated':bool(term.any()),'truncated':bool(trunc.any()),
                                'fired_terms':[n for i,n in enumerate(env.termination_manager._term_names) if bool(env.termination_manager._last_episode_dones[0,i])]}
            save('native_termination.json',native_termination);print('NATIVE_TERMINATION',json.dumps(native_termination),flush=True);break
        assert int(env.episode_length_buf[0])==step+2, 'Unexpected reset'
        post=release_signals_B();raw=bool(rawfunc(env,**rawparams)[0]);banana=env.scene['banana']
        lin=float(torch.linalg.vector_norm(banana.data.root_lin_vel_w[0]).item());ang=float(torch.linalg.vector_norm(banana.data.root_ang_vel_w[0]).item())
        passed=gate.after(int(env.episode_length_buf[0])*float(env.step_dt),post,raw,lin,ang)
        gate_trace.append({'wall_s':time.time(),'step':step+1,'simulation_s':int(env.episode_length_buf[0])*float(env.step_dt),'command':float(predicted[7]),'upstream_predicate':raw,'linear_m_s':lin,'angular_rad_s':ang,'pre':pre,'post':post,'status':gate.status(),'stable_success':bool(passed),'lift_m':lift,'lift_ok':lift_ok})
        postq,postqd,postroot,posttool=state()
        forces=[]
        for side,name in tips.items():
            sensor=env.scene[side+'__banana'];force=sensor.data.force_matrix_w
            assert force is not None and tuple(force.shape)==(1,1,1,3)
            assert list(sensor.body_names)==[name] and sensor.cfg.filter_prim_paths_expr==[cfg.scene.banana.prim_path]
            forces.append(arr(force)[0,0,0])
        objects=np.stack([arr(env.scene[n].data.root_state_w)[0] for n in ('banana','bowl')])
        assert np.isfinite(objects).all() and np.isfinite(forces).all()
        rows.append({'step':step+1,'simulator_step':step+2,'fresh_request':fresh,'chunk_row':logical_step%8,'logical_index_0based':logical_step,'repeat_index_0based':repeat_index,'physical_time_s':(step+1)/15,'logical_elapsed_s':(step+1)/180,'pre_q':q,'pre_tool_root':tool,'q':postq,'qd':postqd,
            'root':postroot,'tool_root':posttool,'virtual_A_q':virtual_seed,'virtual_A_tool':virtual_tool,'state_eef_9d':eef,
            'gripper_observation':closure,'P0_action':predicted,'B_goal':goal,'B_action':arr(action)[0],
            'rate_limited':bool(np.max(np.abs(goal-previous))>.04),'target_B':targetB,
            'tracking_error':pose_error(targetB,posttool),'camera_mapping_error':ce,'virtual_camera_error':vce,
            'pre_camera_world':actualcam,'objects_banana_bowl':objects,'banana_force_on_tips':np.stack(forces),'lift_m':lift,'lift_ok':lift_ok})
        if step%VIDEO_EVERY==0 or passed:
            images=unpack_image_obs(obs);panels=[]
            for label,img in zip(('B EXTERIOR actual','P0 EXTERIOR '+('painted' if args.arm=='painted' else 'actual (painted not sent)'),'WRIST'),(images['over_shoulder_left_camera'],last_painted,images['wrist_cam'])):
                assert img.dtype==np.uint8 and img.ndim==3 and img.shape[2]==3 and img.std()>1
                frame=cv2.resize(img,(640,360));cv2.putText(frame,f'STEP8 {args.policy} {args.variant.upper()} {step+1}/{RUN_TICKS} L{logical_step} {label}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(frame)
            video.write(cv2.hconcat(panels))
        if (step+1)%250==0:print('B_PAINT_DEV_PROGRESS',step+1,json.dumps(gate.status()),flush=True)
        if passed:success_step=step+1;print('GATE_SUCCESS',step+1,flush=True);break
    complete=True
    save('checks.json',{'complete':True,'smoke_non_scoring':SMOKE,'native_termination':native_termination,'arm':args.arm,'variant':args.variant,'scene':args.scene,'policy_seed':POLICY_SEED,'steps':len(rows),'requests':len(capture),'initialization_steps':1,
                        'automatic_v4B_success':success_step is not None,'success_step':success_step,'release_events':gate.events,'final_status':gate.status(),
                        'scope':'Automatic B-ported v4 gate only (the online stop rule); v5 and the step 8 measures are computed afterwards from the saved traces.','policy_label':args.policy,'layout_seed_argument':args.layout_seed})
    print('B_STEP8_COMPLETE',len(rows),'success' if success_step else 'no_success',flush=True)
except BaseException:
    error=traceback.format_exc();(output/'failure.txt').write_text(error);print(error,flush=True)
    raise
finally:
    if rows:np.savez_compressed(output/'tracking.npz',complete=np.asarray(complete),**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    if gate_trace:(output/'gate_trace.jsonl').write_text(''.join(json.dumps(x,default=float)+'\n' for x in gate_trace))
    if trace is not None:trace.close()
    if capture is not None:capture.close()
    if video is not None:video.release()
    if client is not None:client.close()
    if env is not None:env.close()
    app.close()
