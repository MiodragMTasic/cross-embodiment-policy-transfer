"""Predeclared analysis for the painted shadow check (CPU only). Frozen by hash before any painted query exists.

Metric (from shadow_v1's exploratory finding, now pre-registered): P0's action-8 target position, FK of the absolute
joint output, A canonical frame. Per valid row r and cell c, per query i: d_i = |p_i - m_AA(r)|, where m_AA(r) is the
median target of the eight stateA_imgA queries; for stateA_imgA itself the median of the other seven (noise floor).
Row value: median over the eight queries. S_c: mean of row values over the window (first six valid rows).
Banana term: per query, horizontal distance of the target to the banana (world); row value median; D_c = mean over the
window of (row value c - row value stateA_imgA).

Implementation validity (else INVALID; nothing is concluded about P0):
  V1 physics unchanged: at every window row, B's arm joints and banana position match step6_B_shadow_capture_v1
     (same schedule, no painting) within PHYS_Q_RAD and PHYS_BANANA_M.
  V2 painting fidelity: at every window row, mean absolute pixel error (policy-input 180x320) of the painted exterior
     against A2002's recorded exterior is <= FIDELITY_RATIO x that of B's actual exterior.
Fewer than six valid rows -> INSUFFICIENT_VALID_ROWS.
PASS requires all three:
  P1  S_BP <= 2 x S_AA                 (painted request at the noise floor)
  P2  S_BP <= 0.25 x S_BB              (at least 75% of the unpainted shift removed, both measured in this run)
  P3  D_BP <= 15 mm                    (target no longer pushed away from the banana)
Otherwise FAIL, with the failing criteria listed. Bootstrap 95% CIs (resampling queries within cells, 10,000 draws,
seed 0) are reported but do not change the reading. stateA_painted is descriptive."""
import json,sys
import h5py,numpy as np
responses,captures,v1_captures,synchronized,graphs,source=sys.argv[1:7]
sys.path.insert(0,source)
from rigid_chain import RigidChain
# Set from the image-only development probe step6_B_paint_probe_v2 (rows 8-24: B q and banana diff vs v1 exactly 0;
# painted/actual MAE ratio 0.07-0.10) before any painted policy query existed.
PHYS_Q_RAD,PHYS_BANANA_M,FIDELITY_RATIO=1e-5,1e-5,.25
A=RigidChain(json.load(open(graphs))['A'])
C=('stateA_imgA','stateB_imgB','stateB_painted','stateA_painted')
with h5py.File(responses) as f:
    assert f.attrs['complete']
    jp=f['action.joint_position'][:];row=f['meta.row'][:];cond=np.array([c.decode() for c in f['meta.condition'][:]]);rows=f['rows'][:].tolist()
cap=h5py.File(captures);v1=h5py.File(v1_captures);sync=h5py.File(synchronized)
valid={r:bool(cap[str(r)].attrs['valid']) for r in rows}
valid_rows=[r for r in rows if valid[r]];window=valid_rows[:6] if len(valid_rows)>=6 else None
cells={(r,c):np.flatnonzero((row==r)&(cond==c)) for r in rows for c in C}
assert all(len(ix)==8 for ix in cells.values())
world={}
for r in rows:
    g=cap[str(r)];world[r]=(g['A2002_canonical_world'][:]@np.linalg.inv(A.fk(g['A2002_joint_position'][:].astype(float))),g['objects_banana_bowl_table'][0,:3])
P=np.array([A.fk(q[7].astype(float))[:3,3] for q in jp])
def stats(sample):
    S,D={},{}
    for c in C:
        rv,bv=[],[]
        for r in window:
            aa=sample[(r,'stateA_imgA')];ix=sample[(r,c)];W,banana=world[r]
            if c=='stateA_imgA':d=[np.linalg.norm(P[i]-np.median(P[[j for j in aa if j!=i]],axis=0)) for i in ix]
            else:d=[np.linalg.norm(P[i]-np.median(P[aa],axis=0)) for i in ix]
            rv.append(np.median(d))
            tb=lambda idx:np.median([np.linalg.norm((W[:3,:3]@P[i]+W[:3,3])[:2]-banana[:2]) for i in idx])
            bv.append(tb(ix)-tb(aa))
        S[c]=1e3*float(np.mean(rv));D[c]=1e3*float(np.mean(bv))
    return S,D
def mae(a,b):return float(np.abs(a.astype(float)-b.astype(float)).mean())
report={'valid_rows':valid_rows,'window':window}
if window is None:
    report['reading']='INSUFFICIENT_VALID_ROWS'
else:
    checks=[]
    for r in window:
        g=cap[str(r)];o=v1[str(r)];ref=sync['video.exterior_image_1_left'][r]
        checks.append({'row':r,'max_B_q_diff_vs_v1_rad':float(np.abs(g['B_q'][:]-o['B_q'][:]).max()),
                       'banana_diff_vs_v1_m':float(np.linalg.norm(g['objects_banana_bowl_table'][0,:3]-o['objects_banana_bowl_table'][0,:3])),
                       'mae_painted_vs_A2002':mae(g['painted.video.exterior_image_1_left'][0,0],ref),
                       'mae_actual_vs_A2002':mae(g['video.exterior_image_1_left'][0,0],ref),
                       'mae_actual_vs_v1_actual':mae(g['video.exterior_image_1_left'][0,0],o['video.exterior_image_1_left'][0,0])})
    V1=all(c['max_B_q_diff_vs_v1_rad']<=PHYS_Q_RAD and c['banana_diff_vs_v1_m']<=PHYS_BANANA_M for c in checks)
    V2=all(c['mae_painted_vs_A2002']<=FIDELITY_RATIO*c['mae_actual_vs_A2002'] for c in checks)
    S,D=stats(cells)
    rng=np.random.default_rng(0);draws={k:[] for k in ('S_AA','S_BB','S_BP','D_BP','ratio')}
    for _ in range(10000):
        s,d=stats({k:rng.choice(ix,len(ix)) for k,ix in cells.items()})
        for k,v in (('S_AA',s['stateA_imgA']),('S_BB',s['stateB_imgB']),('S_BP',s['stateB_painted']),('D_BP',d['stateB_painted']),('ratio',s['stateB_painted']/s['stateB_imgB'])):draws[k].append(v)
    ci={k:[round(float(x),2) for x in np.percentile(v,[2.5,97.5])] for k,v in draws.items()}
    crit={'P1_S_BP_le_2x_S_AA':S['stateB_painted']<=2*S['stateA_imgA'],'P2_S_BP_le_quarter_S_BB':S['stateB_painted']<=.25*S['stateB_imgB'],'P3_D_BP_le_15mm':D['stateB_painted']<=15}
    report.update({'implementation_checks':checks,'V1_physics_unchanged':V1,'V2_painting_fidelity':V2,
                   'S_mm':{k:round(v,2) for k,v in S.items()},'D_banana_mm':{k:round(v,2) for k,v in D.items()},'bootstrap_ci95':ci,'criteria':crit,
                   'reading':'INVALID' if not (V1 and V2) else 'PASS' if all(crit.values()) else 'FAIL',
                   'failed':[k for k,v in crit.items() if not v]+([] if V1 else ['V1'])+([] if V2 else ['V2'])})
print(json.dumps(report,indent=1,default=bool))
