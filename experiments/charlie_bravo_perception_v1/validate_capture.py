"""Offline evaluation ONLY. Ground truth is never imported by the runtime perception path.
Reads a copied capture run. Returns NO-GO on missing/unvalidated acceptance evidence.
Checks are True, False or None (unknown: matching evidence absent; never substituted).
"""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

MIN_HELD_MEASURED=30      # A4 sample floor
MIN_AVAILABILITY=.80      # measured fraction of held frames, and of hover frames
FUSION_AGREEMENT_M=.03    # candidate per-view agreement rule; provisional, reported only


def summary(errors):
    return {'n':len(errors),'median_m':float(np.median(errors)),'p95_m':float(np.quantile(errors,.95))} if errors else {'n':0}


def bowl_at(run,tick,indices,track):
    """Evaluation-only bowl root at a capture tick. Tick 0 precedes the first tracked step: use startup_after.json."""
    if tick in indices:return track['objects_banana_bowl'][indices[tick]][1,:3]
    startup=run/'startup_after.json'
    if tick==0 and startup.is_file():
        record=json.loads(startup.read_text())
        if record.get('episode_step')==1 and 'bowl' in record.get('objects',{}):return np.asarray(record['objects']['bowl'][:3],float)
    return None


def evaluate(run,reference=None):
    records=[json.loads(p.read_text()) for p in sorted((run/'perception').glob('[0-9]*.json'))]
    if not records:return {'pilot_ready':False,'blockers':['No live RGB-D capture exists.']}
    track=np.load(run/'tracking.npz',allow_pickle=False)
    # Capture tick t is BEFORE step t+1, hence corresponds to post-state of saved step t.
    indices={int(t):i for i,t in enumerate(track['step'])}
    gates={r['step']:r for r in map(json.loads,(run/'gate_trace.jsonl').read_text().splitlines())}
    valid=[];pose=[];times=[];banana_errors=[];bad_offsets=[];hover=[];rows=[];unmatched=[]
    view_errors={'exterior':[],'wrist':[],'agreement_rule':[]};disagreements=0
    held_total=0;held_unknown=Counter();calibration=None
    for r in records:
        for v in r['views'].values():
            pose.append(v['pose_max_error'])
            valid.extend(m['valid_fraction'] for m in v['masks'].values() if m['pixels']>=20 and m['valid_fraction'] is not None)
        times.append(r['capture_seconds'])
        tick=r['tick'];m=r['measurement']
        if calibration is None and m.get('calibration_tick') is not None:calibration=m
        if tick not in indices or tick not in gates:unmatched.append(tick);continue
        objects=track['objects_banana_bowl'][indices[tick]]
        banana,bowl=objects[0,:3],objects[1,:3]
        offset=float(np.linalg.norm(banana[:2]-bowl[:2]))
        post=gates[tick]['post'];held=bool(post['contact'] and not post['table_contact'] and not post['bowl_contact'])
        held_total+=held
        if m.get('status')!='draft_measurement':
            if held:held_unknown[m.get('reason','unknown')]+=1
            if held and offset<=.025:hover.append({'tick':tick,'measured':False,'allowed':False})
            rows.append({'tick':tick,'measured':False,'reason':m.get('reason'),'true_offset_m':offset,'held_evaluation_only':held})
            continue
        error=float(np.linalg.norm(np.asarray(m['banana_surface_median_world_m'])[:2]-banana[:2]))
        per_view={}
        for name in ('exterior','wrist'):
            surface=r['views'][name].get('banana_surface',{})
            if surface.get('points',0)>=40 and surface.get('median_world_m') is not None:
                per_view[name]=np.asarray(surface['median_world_m'])[:2]
        if held:
            banana_errors.append(error)
            for name,xy in per_view.items():view_errors[name].append(float(np.linalg.norm(xy-banana[:2])))
            if len(per_view)==2:
                if np.linalg.norm(per_view['exterior']-per_view['wrist'])<=FUSION_AGREEMENT_M:
                    view_errors['agreement_rule'].append(float(np.linalg.norm((per_view['exterior']+per_view['wrist'])/2-banana[:2])))
                else:disagreements+=1
        allowed=m['geometry_allows_request']
        if allowed and offset>.05:bad_offsets.append(tick)
        if held and offset<=.025:hover.append({'tick':tick,'measured':True,'allowed':allowed})
        rows.append({'tick':tick,'measured':True,'true_offset_m':offset,'held_evaluation_only':held,'geometry_allowed':allowed,'centroid_error_m':error,
                     'lowest_visible_z_minus_root_z_m':m['banana_low_surface_height_above_rim_m']+m['rim_height_m']-banana[2]})
    a1=bool(valid and min(valid)>=.95 and pose and max(pose)<=1e-5)
    # A3 at the calibration instant only; later bowl motion is drift, reported separately.
    calibration_report={'status':'no calibration in capture'};a3=False;drift=None
    if calibration is not None:
        tick=calibration['calibration_tick'];truth=bowl_at(run,tick,indices,track)
        if truth is None:
            calibration_report={'tick':tick,'status':'unknown: no evaluation pose recorded for the calibration tick'};a3=None
        else:
            center_error=float(np.linalg.norm(np.asarray(calibration['bowl_center_xy_m'])-truth[:2]))
            rim_error=float(abs(calibration['rim_height_m']-(truth[2]+.0275)))
            calibration_report={'tick':tick,'status':'measured','center_error_m':center_error,'rim_height_error_m':rim_error}
            a3=bool(center_error<=.005 and rim_error<=.005)
            later=[np.linalg.norm(track['objects_banana_bowl'][indices[t]][1,:2]-truth[:2]) for t in sorted(indices) if t>tick]
            drift=float(max(later)) if later else None
    held_measured=len(banana_errors)
    availability=held_measured/held_total if held_total else None
    hover_availability=sum(x['measured'] for x in hover)/len(hover) if hover else None
    a4=bool(held_measured>=MIN_HELD_MEASURED and availability>=MIN_AVAILABILITY
            and np.median(banana_errors)<=.015 and np.quantile(banana_errors,.95)<=.025)
    a5=bool(hover and hover_availability>=MIN_AVAILABILITY and any(x['allowed'] for x in hover) and not bad_offsets)
    size=sum(p.stat().st_size for p in (run/'perception').rglob('*') if p.is_file())
    a6=bool(times and max(times)<=.5 and size<4_000_000_000)
    comparisons=[];startup=sorted(run.glob('startup_after_*.png'))
    if reference is not None:
        from PIL import Image
        for path in startup:
            other=reference/path.name
            if not other.is_file():comparisons.append({'file':path.name,'missing':True});continue
            a=np.asarray(Image.open(path)).astype(np.int16);b=np.asarray(Image.open(other)).astype(np.int16)
            comparisons.append({'file':path.name,'max_pixel_difference':int(np.abs(a-b).max()) if a.shape==b.shape else None})
    if reference is None or len(startup)<2 or any(x.get('missing') for x in comparisons):
        a2=None;a2_status='unverified: frozen 4003 startup reference images not supplied or incomplete'
    else:
        a2=all(x.get('max_pixel_difference')==0 for x in comparisons);a2_status='compared'
    checks={'A1_sensor_metadata':a1,'A2_startup_image_comparison':a2,'A3_bowl_calibration':a3,'A4_held_banana_centroid':a4,'A5_geometry_veto':a5,'A6_overhead':a6}
    return {'pilot_ready':False,'checks':checks,
            'blockers':[k+('' if v is False else ' (unknown)') for k,v in checks.items() if v is not True]+['Independent capture review and augmented Nemotron prompt replay required.'],
            'startup_image_comparison':{'status':a2_status,'files':comparisons},
            'bowl_calibration':calibration_report,'bowl_drift_after_calibration_max_m':drift,
            'availability':{'held_frames':held_total,'held_measured':held_measured,'held_measured_fraction':availability,
                            'held_unknown_reasons':dict(held_unknown),'hover_frames':len(hover),'hover_measured_fraction':hover_availability,
                            'captures_without_tracking_or_gate_row':unmatched,'minimum_held_measured':MIN_HELD_MEASURED,'minimum_fraction':MIN_AVAILABILITY},
            'per_camera_banana_error':{'pooled_runtime':summary(banana_errors),'exterior_only':summary(view_errors['exterior']),
                                       'wrist_only':summary(view_errors['wrist']),'agreement_rule_mean':summary(view_errors['agreement_rule']),
                                       'agreement_rule_disagreements':disagreements,'agreement_threshold_m':FUSION_AGREEMENT_M,
                                       'status':'PROVISIONAL: runtime still pools both views; choose a rule from native capture data, never from pilot outcomes'},
            'capture_count':len(records),'hover_frames':hover,'unsafe_geometry_approvals':bad_offsets,
            'max_capture_seconds':max(times),'evidence_bytes':size,'geometry_replay_rows':rows,
            'notes':['A2 requires byte-equivalent decoded startup pixels; absent references leave A2 unverified, never passed.',
                     'A3 compares the reference with the bowl pose at the calibration tick; drift is reported, not scored as calibration error.',
                     'Held state, offsets and object poses are evaluation-only and never reach the runtime or prompt.',
                     'Lowest visible point bias is reported relative to banana root, not claimed as physical lowest-point accuracy.',
                     'Missing or unobservable hover prevents pilot approval.']}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--reference',type=Path);args=ap.parse_args()
    result=evaluate(args.run,args.reference);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('geometry_replay_rows','hover_frames')},indent=2))
