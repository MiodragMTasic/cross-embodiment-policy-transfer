"""Generate an isolated auditable runner; never edits or runs the source runner."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'bowl_step8/step8_B_runner.py'


def build(destination=None):
    source = BASE.read_text()
    s = source
    def replace(old, new):
        nonlocal s
        if s.count(old) != 1:
            raise ValueError('source anchor drift: '+old[:70])
        s = s.replace(old, new)
    replace("ROOT = Path.home()/'robot-transfer/bowl'", "HERE = Path(__file__).resolve().parent\nROOT = Path.home()/'robot-transfer/bowl'\nsys.path.insert(0,str(ROOT/'experiments/bowl_step6'))\nsys.path.insert(0,str(HERE))\nfrom runtime import Runtime\nfrom controller import Config, State\nfrom transport import VisionJudge\nfrom vision_images import encode_views")
    replace("parser.add_argument('--case',required=True)", "parser.add_argument('--case',required=True)\nparser.add_argument('--condition',required=True,choices=('baseline','assisted'))\nparser.add_argument('--permit',type=Path,required=True)\nparser.add_argument('--ledger',type=Path,required=True)")
    start = s.index("SMOKE=args.case==")
    end = s.index("assert all(c.isalnum()", start)
    s = s[:start]+'''SMOKE=False
assert args.scene=='layout' and args.policy=='P1' and args.layout_seed==4003 and args.max_ticks is None
assert args.case == f'charlie_B_P1_l4003_{args.condition}_v1'
# Fail before launching Isaac if review/account/budget permit is absent.
judge=VisionJudge(args.permit,args.ledger)
judge.verify_source_lock(HERE/'source-lock.json')
'''+s[end:]
    replace("RUN_TICKS=args.max_ticks if SMOKE else TICKS", "RUN_TICKS=3600  # both conditions: 240 simulated seconds, or existing success/safety stop")
    replace("native_termination=None;lift=0.;closed_run=0;lift_ok=False;post=None", "native_termination=None;lift=0.;closed_run=0;lift_ok=False;post=None\n    drop=Runtime(output/'drop',judge,enabled=args.condition=='assisted',config=Config(max_calls=RUN_TICKS//12+2*3,max_wall_s=2400,api_cap=1.0,reserve_per_call=.003))\n    force_policy_request=False\n    save('drop_protocol.json',{'condition':args.condition,'max_ticks':RUN_TICKS,'layout':4003,'policy':'P1','scope':'selected-case paused-simulation mechanism demo; not original step8; one pair is not a causal counterfactual','primary_outcome':'offline v5.1 first stable rest on executed command; also report pass_at_end and post-DONE contact count','judge_exhaustion':'after three failed confirmations continue policy-only and retain episode in denominator','termination':'same v4 success, native/safety, 3600 ticks or wall deadline','source_sha256':digest(Path(__file__))})")
    replace("fresh=step%96==0", "finishing_open=(drop.controller.state==State.OPEN and step-drop.controller.open_tick>=drop.controller.config.open_ticks)\n        intervention=drop.controller.current_override() is not None and not finishing_open\n        fresh=(step%96==0 or force_policy_request or finishing_open) and not intervention\n        if fresh: force_policy_request=False")
    start=s.index("        predicted=chunk[logical_step%8]")
    end=s.index("        obs,_,term,trunc,_=env.step(action)",start)
    old=s[start:end]
    mapping=old[:old.index("        action=torch.as_tensor")]
    # Move policy mapping behind the intervention decision. B mapping and lift state
    # are never advanced by unused policy commands while a pose is held.
    mapping=mapping.replace("        predicted=chunk[logical_step%8];Aq=predicted[:7].astype(float)\n", "")
    mapping='\n'.join('    '+line if line else line for line in mapping.rstrip().split('\n'))
    new='''        predicted=chunk[logical_step%8]
        def capture_drop():
            # Current actual images; never last_painted. No extra env.step/render.
            return encode_views(images['over_shoulder_left_camera'], images['wrist_cam'])
        was_hold=drop.controller.current_override() is not None
        proposal=drop.step(step,tuple(q[ids]),tuple(command),float(predicted[7]),capture_drop,record_command=False)
        if proposal.stop:
            native_termination={'step':step,'reason':proposal.reason,'scope':'DROP budget/deadline stop'}
            break
        override=drop.controller.current_override()
        if was_hold and override is None:
            force_policy_request=not fresh
            override=(tuple(command),0.0 if drop.controller.used else float(predicted[7]))  # one tick hold; fresh P1 request before resuming mapping
        if override is not None:
            previous=command.copy();goal=np.asarray(override[0]);targetB=tool.copy()
            command+=np.clip(goal-command,-.04,.04)
            executed_grip=override[1]
        else:
            Aq=predicted[:7].astype(float)
'''+mapping+'''
            executed_grip=0.0 if drop.controller.used else float(predicted[7])
        assert np.max(np.abs(command-previous))<=.04000001
        assert np.isfinite(command).all() and (command>=Blim[:,0]).all() and (command<=Blim[:,1]).all()
        action=torch.as_tensor(np.r_[command,executed_grip],device=env.device,dtype=torch.float32)[None]
        drop.log({'tick':step,'policy_A':predicted.tolist(),'executed_B':arr(action)[0].tolist(),'lift_m':lift,'override':override is not None})
        pre=release_signals_B();t_pre=int(env.episode_length_buf[0])*float(env.step_dt);gate.before(t_pre,float(executed_grip),pre)
'''
    s=s[:start]+new+s[end:]
    replace("'command':float(predicted[7]),'upstream_predicate'", "'command':float(executed_grip),'policy_command':float(predicted[7]),'upstream_predicate'")
    replace("f'STEP8 {args.policy} {args.variant.upper()}", "f'CHARLIE {args.policy} {args.condition.upper()}")
    replace("        predicted=chunk[logical_step%8]", "        predicted=chunk[((step-chunk_origin_tick)//12)%8]")
    replace("chunk=client._postprocess_chunk(client._unpack_response(response));assert chunk.shape==(40,8)", "chunk=client._postprocess_chunk(client._unpack_response(response));assert chunk.shape==(40,8)\n            chunk_origin_tick=step")
    replace("'chunk_row':logical_step%8", "'chunk_row':((step-chunk_origin_tick)//12)%8")
    replace("'max_steps':TICKS", "'max_steps':RUN_TICKS")
    replace("'total_simulator_steps':TICKS+1", "'total_simulator_steps':RUN_TICKS+1")
    replace("Step 8 final-test episode on a reserved layout (or the declared non-scoring smoke check):", "Charlie Bravo exploratory selected-case episode, separate from the frozen Step 8 test:")
    replace("Never used for tuning.", "Known failure selected for development; not held-out evidence.")
    replace("digest(Path(__file__).with_name(n))", "digest(ROOT/'experiments/bowl_step6'/n)")
    replace("judge=VisionJudge(args.permit,args.ledger)", "judge=VisionJudge(args.permit,args.ledger,require_key=args.condition=='assisted')")
    replace("judge.verify_source_lock(HERE/'source-lock.json')", "judge.verify_source_lock(HERE/'source-lock.json')\nfor helper in ('run_painted_dev.py','paint_A.py','local_ik.py','rigid_chain.py'):\n    if not (ROOT/'experiments/bowl_step6'/helper).is_file(): raise FileNotFoundError(helper)")
    replace("                if predicted[7]>.5:", "                if predicted[7]>.5 and not drop.controller.used:")
    replace("                else:closed_run=0", "                else:closed_run=0;lift_ok=False")
    replace("    complete=True\n    save('checks.json',{'complete':True", "    complete=not (native_termination and native_termination.get('scope')=='DROP budget/deadline stop')\n    save('checks.json',{'complete':complete,'valid_for_comparison':complete")
    replace("print('B_STEP8_COMPLETE',", "print('CHARLIE_COMPLETE' if complete else 'CHARLIE_INVALID',")
    replace("save('inventory.json',inventory)", "inventory['drop_override']={'condition':args.condition,'hold_target':'last executed B command','confirm_after_controls':9,'open_arm_hold_controls':90,'gripper':'permanently open after confirmed DROP; otherwise P1','max_failed_holds':3,'failed_holds':'continue policy and count','vision_only':True,'crop':'fixed exterior crop; full wrist','fresh_policy':'96 controls plus resume after intervention'}\n    save('inventory.json',inventory)")
    ast.parse(s)
    output_dir=Path(destination) if destination is not None else HERE
    output_dir.mkdir(parents=True,exist_ok=True)
    dest=output_dir/'runner_B.py'
    dest.write_text(s)
    (output_dir/'runner-provenance.json').write_text(json.dumps({'base':str(BASE.relative_to(HERE.parents[1])), 'base_sha256':hashlib.sha256(source.encode()).hexdigest(),'generated_sha256':hashlib.sha256(s.encode()).hexdigest(),'note':'Review generated diff; static compilation is not simulator validation.'},indent=2)+'\n')
    return dest

if __name__=='__main__': print(build())
