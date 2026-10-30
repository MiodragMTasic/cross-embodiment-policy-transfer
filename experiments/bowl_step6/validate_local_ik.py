"""Predeclared CPU IK bank, singular-start and impossible-target controls; no policy."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
from rigid_chain import RigidChain,pose_matrix,rigid_inverse
from local_ik import solve,physical_limits,pose_error

parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
base=args.workspace/'assets/report/evidence/2026-10-21/bowl-step6'
inputs={'graph':base/'kinematics_export_v1/joint_graphs.json','recorded':base/'recorded_fk_v1/pose_comparisons.npz'}
results=[]
try:
    graphs=json.loads(inputs['graph'].read_text());recorded=np.load(inputs['recorded']);rng=np.random.default_rng(621)
    # SO(3) numerical edge cases: finite near-zero and pi errors with correct magnitudes.
    for theta in (0.,1e-10,np.pi-1e-10,np.pi):
        t=pose_matrix([0,0,0],[np.cos(theta/2),0,0,np.sin(theta/2)])
        e=pose_error(t,np.eye(4));assert np.isfinite(e).all() and abs(np.linalg.norm(e[3:])-theta)<1e-8
    bank=[]
    for label in ('A','B'):
        model=RigidChain(graphs[label]);limits=physical_limits(graphs[label]);qs=recorded[label+'_q']
        for idx in np.linspace(0,len(qs)-1,7,dtype=int):
            q=qs[idx].astype(float)
            seed=np.clip(q+rng.uniform(-.15,.15,7),limits[:,0]+1e-5,limits[:,1]-1e-5)
            assert np.linalg.norm(seed-q)>.01
            bank.append({'name':f'{label}_recorded_{idx}','robot':label,'seed':seed.tolist(),'target':model.fk(q).tolist(),'expected':'converge','tool_offset':None})
        # Offset target exercises TCP pose, with the same physical arm.
        offset=pose_matrix([.02,.03,.04],[np.sqrt(.5),0,0,np.sqrt(.5)])
        q=qs[0].astype(float);seed=np.clip(q+np.array([.1,-.1,.1,-.1,.1,-.1,.1]),limits[:,0]+1e-5,limits[:,1]-1e-5)
        bank.append({'name':f'{label}_tool_offset','robot':label,'seed':seed.tolist(),'target':model.fk(q,offset).tolist(),'expected':'converge','tool_offset':offset.tolist()})
        if label=='B':
            seed=np.zeros(7);q=np.array([.04,.06,-.04,.04,.03,-.05,.04])
            singular=np.linalg.svd(model.fk_jacobian(seed)[1],compute_uv=False)
            assert singular[-1]<1e-8, singular
            bank.append({'name':'B_singular_start','robot':label,'seed':seed.tolist(),'target':model.fk(q).tolist(),'expected':'converge','tool_offset':None,'seed_jacobian_singular_values':singular.tolist()})
        unreachable=np.eye(4);unreachable[:3,3]=[5.,5.,5.]
        bank.append({'name':f'{label}_impossible_target','robot':label,'seed':qs[0].tolist(),'target':unreachable.tolist(),'expected':'fail','tool_offset':None})
    # Persist all targets, seeds, expectations and source identities before solving any case.
    provenance={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in inputs.items()}
    for name in ('rigid_chain.py','local_ik.py','validate_local_ik.py'):
        p=Path(__file__).with_name(name);provenance[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (args.output/'bank.json').write_text(json.dumps({'random_seed':621,'max_iterations':200,'position_tolerance_m':1e-4,'rotation_tolerance_rad':1e-3,'provenance':provenance,'cases':bank},indent=2)+'\n')
    for case in bank:
        model=RigidChain(graphs[case['robot']]);limits=physical_limits(graphs[case['robot']])
        offset=None if case['tool_offset'] is None else np.asarray(case['tool_offset'])
        target=np.asarray(case['target']);result=solve(model,target,np.asarray(case['seed']),limits,offset)
        result.update(name=case['name'],expected=case['expected']);results.append(result)
        (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        assert result['success']==(case['expected']=='converge'),(case['name'],result['reason'],result['position_error_m'],result['rotation_error_rad'])
        q=np.asarray(result['q']);assert np.isfinite(q).all() and (q>=limits[:,0]).all() and (q<=limits[:,1]).all()
        actual=pose_error(target,model.fk(q,offset))
        assert np.allclose([np.linalg.norm(actual[:3]),np.linalg.norm(actual[3:])],[result['position_error_m'],result['rotation_error_rad']],atol=1e-12)
        assert result['iterations']<=200 and result['fk_evaluations']<=1201
    (args.output/'checks.json').write_text(json.dumps({'complete':True,'cases':len(results),'converged':sum(r['success'] for r in results),'failed_as_expected':sum(not r['success'] for r in results),'scope':'CPU local IK only; no physical tracking or collision-free motion claim.'},indent=2)+'\n')
    print('CPU_IK_BANK_COMPLETE',len(results))
except BaseException:
    (args.output/'failure.txt').write_text(traceback.format_exc());raise
