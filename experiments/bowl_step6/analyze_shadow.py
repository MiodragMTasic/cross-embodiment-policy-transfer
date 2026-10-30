"""Predeclared analysis for the matched shadow test (CPU only). Frozen by hash before any data exists.

Metric per query: tilt8 = tilt(action 8) - tilt(sent state), the tilt P0 adds over one executed segment of 8 actions,
tilt measured as the angle between A's canonical approach axis and straight down, using FK of the absolute joint output.
Per valid request row and condition: median over repeats. Effects relative to stateA_imgA at the same row:
  image  = stateA_imgB - stateA_imgA   (B images, A2002 state)             primary
  state  = stateB_imgA - stateA_imgA   (B mapped state, A2002 images)      primary
  total  = stateB_imgB - stateA_imgA   (everything B would send)           primary
  wrist  = stateA_wristB - stateA_imgA (B wrist image only)                secondary
  exterior = stateA_extB - stateA_imgA (B exterior image only)             secondary
Primary window: the first six valid request rows; fewer than six valid rows -> INSUFFICIENT_VALID_ROWS for every effect.
Secondary window: rows 8, 16, 24 (where the live excess concentrated); all three must be valid, else INSUFFICIENT_VALID_ROWS.
Statistic: mean effect over the window; 95% bootstrap CI resampling repeats within each row/condition (10,000 draws, seed 0).
Reading per effect: SUPPORTED if mean >= 4 deg and CI lower > 2 deg; RULED OUT if CI upper < 2 deg; else INCONCLUSIVE.
Live reference for scale: over requests 2-4, B added 7.4-11.4 deg per 8 actions where A added -3.0 to 6.2 deg.
Scope: one scene (A2002 layout); the live B runs used layout 0. A RULED OUT total holds for this scene only."""
import json,sys
import h5py,numpy as np
responses,captures,graphs,source=sys.argv[1:5]
sys.path.insert(0,source)
from rigid_chain import RigidChain
A=RigidChain(json.load(open(graphs))['A'])
def tilt(q):return float(np.degrees(np.arccos(np.clip(-A.fk(np.asarray(q,float))[:3,0][2],-1,1))))
def fkpose(q):return A.fk(np.asarray(q,float))
with h5py.File(captures) as f:valid={int(k):bool(f[k].attrs['valid']) for k in f};checks={int(k):dict(f[k].attrs) for k in f}
with h5py.File(responses) as f:
    assert f.attrs['complete']
    jp=f['action.joint_position'][:];state=f['sent.state.joint_position'][:]
    row=f['meta.row'][:];cond=np.array([c.decode() for c in f['meta.condition'][:]])
    rows=f['rows'][:].tolist();recorded=f['A2002_recorded_first8'][:]
C=('stateA_imgA','stateB_imgB','stateA_imgB','stateB_imgA','stateA_wristB','stateA_extB')
tilt8=np.array([tilt(j[7])-tilt(s) for j,s in zip(jp,state)]);tilt1=np.array([tilt(j[0])-tilt(s) for j,s in zip(jp,state)])
cells={(r,c):np.flatnonzero((row==r)&(cond==c)) for r in rows for c in C}
assert all(len(ix)==8 for ix in cells.values())
valid_rows=[r for r in rows if valid[r]]
WINDOWS={'primary':valid_rows[:6] if len(valid_rows)>=6 else None,
         'secondary_rows_8_24':[8,16,24] if all(valid.get(r) for r in (8,16,24)) else None}
EFFECTS={'image':'stateA_imgB','state':'stateB_imgA','total':'stateB_imgB','wrist':'stateA_wristB','exterior':'stateA_extB'}
def effects(sample,window):
    med={key:np.median(tilt8[ix]) for key,ix in sample.items()}
    return {e:float(np.mean([med[(r,c)]-med[(r,'stateA_imgA')] for r in window])) for e,c in EFFECTS.items()}
def reading(m,lo,hi):return 'SUPPORTED' if m>=4 and lo>2 else 'RULED OUT' if hi<2 else 'INCONCLUSIVE'
def interpret(s):
    t,i,st=(s[e]['reading'] for e in ('total','image','state'))
    if 'INSUFFICIENT_VALID_ROWS' in (t,i,st):return 'insufficient valid rows; no reading'
    if t=='RULED OUT':return 'matched-state B observations do not reproduce the live excess in this scene'
    if t!='SUPPORTED':return 'inconclusive: matched-state reproduction of the live excess is unresolved'
    if i=='SUPPORTED' and st=='SUPPORTED':return 'both B images and B state mapping contribute'
    if i=='SUPPORTED':return 'B images drive the matched-state excess'
    if st=='SUPPORTED':return 'B state mapping drives the matched-state excess'
    return 'interaction or weak parts: total reproduced, but neither images nor state alone is supported'
report={}
for name,window in WINDOWS.items():
    if window is None:
        report[name]={'rows':None,'effects':{e:{'reading':'INSUFFICIENT_VALID_ROWS'} for e in EFFECTS}}
        report[name]['interpretation']=interpret(report[name]['effects']);continue
    point=effects(cells,window);rng=np.random.default_rng(0);draws={e:[] for e in EFFECTS}
    for _ in range(10000):
        for e,v in effects({key:rng.choice(ix,len(ix)) for key,ix in cells.items()},window).items():draws[e].append(v)
    summary={}
    for e in EFFECTS:
        lo,hi=np.percentile(draws[e],[2.5,97.5]);summary[e]={'mean_deg':round(point[e],2),'ci95_deg':[round(float(lo),2),round(float(hi),2)],'reading':reading(point[e],lo,hi)}
    report[name]={'rows':window,'effects':summary,'interpretation':interpret(summary)}
# Descriptive: reproducibility of the exact recorded A2002 request, and full canonical-pose shift at action 8.
repro=[]
for i,r in enumerate(rows):
    ix=cells[(r,'stateA_imgA')]
    repro.append({'row':r,'max_joint_diff_first8_vs_recorded_rad':round(float(np.max([np.abs(jp[j,:8]-recorded[i][:,:7]).max() for j in ix])),5)})
per_row=[]
for r in rows:
    ref=fkpose(np.median(jp[cells[(r,'stateA_imgA')],7],axis=0))
    entry={'row':r,'valid':valid[r],**{k:round(float(v),5) for k,v in checks[r].items() if k.endswith(('_m','_rad'))}}
    for c in C:
        ix=cells[(r,c)];pose=fkpose(np.median(jp[ix,7],axis=0))
        entry[c]={'tilt8_median':round(float(np.median(tilt8[ix])),2),'tilt8_range':[round(float(tilt8[ix].min()),2),round(float(tilt8[ix].max()),2)],
                  'tilt1_median':round(float(np.median(tilt1[ix])),2),
                  'action8_position_shift_vs_AA_mm':round(1e3*float(np.linalg.norm(pose[:3,3]-ref[:3,3])),2),
                  'action8_rotation_shift_vs_AA_deg':round(float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@ref[:3,:3])-1)/2,-1,1)))),2)}
    per_row.append(entry)
print(json.dumps({'valid_rows':valid_rows,'excluded_rows':[r for r in rows if not valid[r]],**report,
                  'A2002_recorded_reproducibility':repro,'per_row':per_row},indent=1))
