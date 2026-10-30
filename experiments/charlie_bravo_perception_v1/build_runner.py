"""Generate isolated capture/pilot runner for review; no Isaac execution."""
import ast,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
BASE=P.parent/'charlie_bravo_v1/runner_B.py'
def build(destination=None):
 s=BASE.read_text()
 def replace(old,new):
  nonlocal s
  if s.count(old)!=1:raise ValueError('anchor drift: '+old[:80])
  s=s.replace(old,new)
 replace('sys.path.insert(0,str(HERE))', "sys.path.insert(0,str(HERE.parent/'charlie_bravo_v1'))\nsys.path.insert(0,str(HERE))\nfrom bridge import Collector, PerceptionJudge")
 replace("parser.add_argument('--condition',required=True,choices=('baseline','assisted'))", "parser.add_argument('--condition',required=True,choices=('baseline','assisted'))\nparser.add_argument('--mode',required=True,choices=('capture','pilot'))")
 replace("assert args.case == f'charlie_B_P1_l4003_{args.condition}_v1'", "assert args.case == f'charlie_depth_B_P1_l4003_{args.mode}_v1'\nassert args.condition == ('baseline' if args.mode=='capture' else 'assisted')")
 replace("judge=VisionJudge(args.permit,args.ledger,require_key=args.condition=='assisted')", "from preflight import CaptureJudge, check\nlaunch_check=check(json.loads(args.permit.read_text()))\nif not launch_check['launch_ready']:raise PermissionError(launch_check['blockers'])\njudge=CaptureJudge(args.permit) if args.mode=='capture' else VisionJudge(args.permit,args.ledger,require_key=True)")
 replace("judge.verify_source_lock(HERE/'source-lock.json')", "judge.verify_source_lock(HERE/'source-lock.json')\nfrom layout_pin import verify_layout\nverify_layout(HERE.parents[1])\nif args.mode=='pilot' and judge.permit.get('perception_review_status')!='OK':\n    raise PermissionError('perception pilot not reviewed')")
 replace('RUN_TICKS=3600  # both conditions: 240 simulated seconds, or existing success/safety stop', 'RUN_TICKS=3600  # capture baseline and pilot: 240s, never the frozen 600s test')
 replace("    add_painted_A(cfg)", "    for name in ('wrist_cam','over_shoulder_left_camera'):\n        camera_cfg=getattr(cfg.scene,name)\n        camera_cfg.data_types=list(dict.fromkeys([*camera_cfg.data_types,'distance_to_image_plane']))\n    add_painted_A(cfg)")
 replace("    force_policy_request=False", "    depth_capture=Collector(output/'perception')\n    drop.judge=PerceptionJudge(judge,depth_capture,validated=judge.permit.get('perception_review_status')=='OK')\n    save('perception_protocol.json',{'mode':args.mode,'scoring':'240s sensor baseline; offline v5.1 only' if args.mode=='capture' else 'selected-case240s exploratory pilot','extra_sensor':'ideal RGB-D depth','ground_truth_object_inputs':False,'P1_inputs':'unchanged RGB/state','geometry':'early exterior bowl fit; merged visible banana centroid veto','hold_and_open':'existing Charlie controller'})\n    force_policy_request=False")
 replace("        images=unpack_image_obs(obs)\n        closure=", """        images=unpack_image_obs(obs)
        # Capture before painter.paint: force no extra render or physics step.
        if step-drop.last_capture >= (3 if drop.controller.state==State.HOLD else 6):
            views={}
            for label,name in (('exterior','over_shoulder_left_camera'),('wrist','wrist_cam')):
                camera=env.scene[name];data=camera.data
                native=arr(data.output['rgb'])[0,:,:,:3]
                if not np.array_equal(native,images[name]):raise ValueError('RGB observation/sensor mismatch')
                views[label]={'rgb':native.copy(),'depth':arr(data.output['distance_to_image_plane'])[0].copy(),
                    'K':arr(data.intrinsic_matrices)[0].copy(),
                    'world_from_opengl':pose_matrix(arr(data.pos_w)[0],arr(data.quat_w_opengl)[0]),
                    'world_from_ros':pose_matrix(arr(data.pos_w)[0],arr(data.quat_w_ros)[0]),
                    'frame_counter':int(arr(camera.frame).reshape(-1)[0])}
            depth_capture.capture(step,views)
        closure=""")
 ast.parse(s)
 dest=Path(destination) if destination else P;dest.mkdir(parents=True,exist_ok=True)
 (dest/'runner_B.py').write_text(s)
 (dest/'runner-provenance.json').write_text(json.dumps({'base':str(BASE),'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(s.encode()).hexdigest(),'live_tested':False},indent=2))
 return s
if __name__=='__main__':build()
