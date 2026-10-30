"""Stage 2 of the painted shadow check: server-only queries on saved inputs. Nothing is executed.
Cells: A2002 recorded request; B unpainted; B with the painted exterior (what the painted bridge sends);
A2002 state and wrist with the painted exterior (painting fidelity in policy terms)."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
ROOT=Path.home()/'robot-transfer/bowl';REPO=ROOT/'RoboLab'
sys.path.insert(0,str(REPO))
import h5py,numpy as np,zmq
from policies.gr00t.client import GR00TPolicyClient,_get_action_value,_as_action_chunk
p=argparse.ArgumentParser();p.add_argument('--case',required=True);args=p.parse_args()
assert args.case=='step6_B_paint_query_v1'
out=ROOT/'runs'/args.case;out.mkdir(parents=True,exist_ok=False)
CAPTURE=ROOT/'runs/step6_B_paint_capture_v1/captures.hdf5'
SYNC=REPO/'output/step5_training_layout2002_v4/BananaInBowlUniformInitPose10cmTask/synchronized.hdf5'
SYNC_SHA='0d2096b1939d15ff086c1c7de8c6ce638f5a29ed5ab46a7e2942ecf5cd9eafc1'
REPEATS=8
# name: (state source, exterior image source, wrist image source); P = painted exterior from stage 1
SOURCES={'stateA_imgA':'AAA','stateB_imgB':'BBB','stateB_painted':'BPB','stateA_painted':'APA'}
CONDITIONS=tuple(SOURCES)
VIDEO=('video.exterior_image_1_left','video.wrist_image_left');STATE=('state.eef_9d','state.joint_position','state.gripper_position')
client=None;results=None
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
    assert digest(SYNC)==SYNC_SHA
    with h5py.File(CAPTURE) as f:
        instruction=f.attrs['instruction'];rows=sorted(int(k) for k in f)
        B={r:{k:f[str(r)][k][:] for k in VIDEO+STATE} for r in rows}
        P={r:{'video.exterior_image_1_left':f[str(r)]['painted.video.exterior_image_1_left'][:]} for r in rows}
        valid={r:bool(f[str(r)].attrs['valid']) for r in rows}
    with h5py.File(SYNC) as f:
        assert f.attrs['instruction']==instruction
        A={r:{k:f[k][r][None,None] for k in VIDEO+STATE} for r in rows}
        recorded={r:f['action.executed_joint_gripper'][r:r+8] for r in rows}
    assert rows==list(range(8,89,8)),rows
    for r in rows:
        for k in VIDEO+STATE:assert A[r][k].shape==B[r][k].shape and A[r][k].dtype==B[r][k].dtype,(r,k)
        k='video.exterior_image_1_left';assert P[r][k].shape==A[r][k].shape and P[r][k].dtype==A[r][k].dtype,(r,'painted')
    def request(r,condition):
        state,exterior,wrist=({'A':A,'B':B,'P':P}[x] for x in SOURCES[condition])
        return {'video.exterior_image_1_left':exterior[r]['video.exterior_image_1_left'],'video.wrist_image_left':wrist[r]['video.wrist_image_left'],
                **{k:state[r][k] for k in STATE},'annotation.language.language_instruction':[instruction]}
    # Fixed order, rotated across conditions within each (repeat, row) block; one server RNG stream.
    order=[(k,r,CONDITIONS[(j+k+n)%len(CONDITIONS)]) for k in range(REPEATS) for j,r in enumerate(rows) for n in range(len(CONDITIONS))]
    (out/'protocol.json').write_text(json.dumps({'scope':'Server-only policy queries on saved inputs. No simulation, no action executed, no score.',
        'capture':str(CAPTURE),'capture_sha256':digest(CAPTURE),'synchronized_sha256':SYNC_SHA,'rows':rows,'valid_rows':[r for r in rows if valid[r]],
        'conditions':{'stateA_imgA':'A2002 recorded request (control and noise floor)','stateB_imgB':'unpainted B request (baseline)',
                      'stateB_painted':'B state, painted exterior, B wrist (what the painted bridge sends)','stateA_painted':'A2002 state and wrist, painted exterior (painting fidelity)'},
        'repeats':REPEATS,'queries':len(order),'policy_seed_env':'POLICY_SEED=0, seeded once at server start','instruction':instruction},indent=2)+'\n')
    client=GR00TPolicyClient(host='127.0.0.1',port=15555)
    for o in (zmq.RCVTIMEO,zmq.SNDTIMEO):client.socket.setsockopt(o,60000)
    client.socket.setsockopt(zmq.LINGER,0)
    results=h5py.File(out/'responses.hdf5','x');results.attrs['instruction']=instruction
    keep={'joint_position':(40,7),'gripper_position':(40,1),'eef_9d':(40,9)}
    store={n:results.create_dataset('action.'+n,shape=(len(order),*s),dtype='f4') for n,s in keep.items()}
    sent={k:results.create_dataset('sent.'+k,shape=(len(order),*A[rows[0]][k].shape[2:]),dtype='f4') for k in STATE}
    meta={'order':[],'repeat':[],'row':[],'condition':[]}
    for i,(k,r,condition) in enumerate(order):
        req=request(r,condition);response=client.get_action(req)
        for n,s in keep.items():
            chunk=_as_action_chunk(_get_action_value(response[0],n),name=n);assert chunk.shape==s and np.isfinite(chunk).all(),(n,chunk.shape)
            store[n][i]=chunk
        for key in STATE:sent[key][i]=req[key][0,0]
        for name,value in zip(meta,(i,k,r,condition)):meta[name].append(value)
        if (i+1)%32==0:results.flush();print('SHADOW_QUERY_PROGRESS',i+1,len(order),flush=True)
    for name,value in meta.items():results.create_dataset('meta.'+name,data=np.asarray(value,dtype='S16' if name=='condition' else 'i4'))
    results.create_dataset('A2002_recorded_first8',data=np.stack([recorded[r] for r in rows]));results.create_dataset('rows',data=np.asarray(rows,dtype='i4'))
    results.attrs['complete']=True
    print('SHADOW_QUERY_COMPLETE',len(order),flush=True)
except BaseException:
    error=traceback.format_exc();(out/'failure.txt').write_text(error);print(error,flush=True)
    raise
finally:
    if results is not None:results.close()
    if client is not None:client.close()
