"""Native sensor evidence and a measured veto; object-state inputs are forbidden."""
import io,json,time
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from perception import masks,backproject,camera_to_world,BowlReference,largest_component
sys.path.append(str(Path(__file__).resolve().parents[1]/'charlie_bravo_v1'))
from controller import Decision


def optical_from_opengl(world_from_opengl):
    return np.asarray(world_from_opengl) @ np.diag([1.,-1.,-1.,1.])


def prompt_measurements(evidence):
    keys=('banana_surface_offset_m','banana_low_surface_height_above_rim_m',
          'banana_points','bowl_points','bowl_surface_mismatch_m','bowl_still_matches')
    return [None if e is None else {'tick':e['tick'],**{k:e.get('measurement',{}).get(k) for k in keys}} for e in evidence]


class Collector:
    def __init__(self,path,byte_cap=4_000_000_000):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False)
        self.records={};self.bowl=BowlReference();self.bytes_written=0
        self.capture_count=0
        self.byte_cap=byte_cap;self.last_frames={};self.last_tick=-1

    def capture(self,tick,views):
        start=time.monotonic()
        if tick<=self.last_tick:raise ValueError('non-increasing capture tick')
        arrays={};reports={};all_points={};files={}
        for name in ('exterior','wrist'):
            v=views[name];rgb=np.asarray(v['rgb']);depth=np.asarray(v['depth'])
            if depth.shape==rgb.shape[:2]+(1,):depth=depth[:,:,0]
            if depth.shape!=rgb.shape[:2]:raise ValueError('RGB-depth shape mismatch')
            counter=v['frame_counter']
            if type(counter) is not int or counter<=self.last_frames.get(name,-1):raise ValueError('stale camera frame')
            t=optical_from_opengl(v['world_from_opengl'])
            if not np.allclose(t,v['world_from_ros'],atol=1e-5,rtol=0):raise ValueError('camera pose convention mismatch')
            segments=masks(rgb);valid=np.isfinite(depth)&(depth>0)&(depth<10)
            points={obj:camera_to_world(backproject(depth,v['K'],mask),t) for obj,mask in segments.items()}
            all_points[name]=points
            if name=='exterior' and self.bowl.reference is None and tick<=60:
                calibration=camera_to_world(backproject(depth,v['K'],largest_component(segments['bowl'])),t)
            reports[name]={'frame_counter':counter,'depth_dtype':str(depth.dtype),'depth_shape':list(depth.shape),
                'pose_max_error':float(np.max(np.abs(t-v['world_from_ros']))),
                'depth_min_m':float(depth[valid].min()) if valid.any() else None,
                'depth_max_m':float(depth[valid].max()) if valid.any() else None,
                'masks':{obj:{'pixels':int(mask.sum()),'valid_fraction':float(valid[mask].mean()) if mask.any() else None} for obj,mask in segments.items()}}
            banana=points['banana']
            reports[name]['banana_surface']={'points':len(banana),
                'median_world_m':np.median(banana,axis=0).tolist() if len(banana) else None,
                'low_z_p05_m':float(np.quantile(banana[:,2],.05)) if len(banana) else None}
            buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='PNG')
            files[f'{tick:05d}_{name}.png']=buffer.getvalue()
            # Millimetre quantization <=0.5 mm; 0 explicitly represents invalid depth.
            depth_mm=np.zeros(depth.shape,np.uint16);depth_mm[valid]=np.maximum(1,np.rint(depth[valid]*1000)).astype(np.uint16)
            arrays.update({f'{name}_depth_mm':depth_mm,f'{name}_K':v['K'],f'{name}_world_from_optical':t,
                           f'{name}_mask_shape':np.asarray(depth.shape)})
            arrays.update({f'{name}_{obj}_packed':np.packbits(mask) for obj,mask in segments.items()})
        measurement=self.bowl.measure(tick,np.concatenate([all_points[n]['banana'] for n in ('exterior','wrist')]),all_points['exterior']['bowl'],calibration_world=calibration if self.bowl.reference is None and tick<=60 else None)
        record={'tick':tick,'views':reports,'measurement':measurement,
                'geometry_allows_request':measurement['geometry_allows_request'],
                'status':'UNVALIDATED_DEVELOPMENT_MEASUREMENT','depth_encoding':'uint16 millimetres, zero invalid'}
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);files[f'{tick:05d}.npz']=buffer.getvalue()
        # Reserve space for JSON before writing anything from this capture.
        projected=self.bytes_written+sum(map(len,files.values()))+16384
        if projected>self.byte_cap:raise RuntimeError('capture evidence cap reached')
        for filename,data in files.items():(self.path/filename).write_bytes(data)
        self.capture_count+=1
        if self.capture_count==10:
            record['storage_projection']={'captures':600,'bytes':int(projected/self.capture_count*600),
                'abort_threshold_bytes':min(3_500_000_000,int(self.byte_cap*.875))}
        record['capture_seconds']=time.monotonic()-start
        payload=json.dumps(record,allow_nan=False,indent=2).encode()
        if len(payload)>16384:raise RuntimeError('capture metadata unexpectedly large')
        (self.path/f'{tick:05d}.json').write_bytes(payload)
        self.bytes_written+=sum(map(len,files.values()))+len(payload)
        self.records[tick]=record;self.last_tick=tick
        self.last_frames={name:views[name]['frame_counter'] for name in views}
        for old in sorted(self.records)[:-12]:del self.records[old]
        forecast=record.get('storage_projection')
        if forecast and forecast['bytes']>forecast['abort_threshold_bytes']:
            raise RuntimeError('capture storage forecast exceeds limit; first ten captures preserved')
        return record


class PerceptionJudge:
    def __init__(self,judge,collector,validated=False):
        self.judge=judge;self.collector=collector;self.validated=validated

    def __call__(self,prompt,frames,request):
        tick=request['tick'];evidence=[self.collector.records.get(f.tick) for f in frames]
        current=bool(evidence and frames[-1].tick==tick and all(e is not None for e in evidence))
        permits=bool(self.validated and current and all(e['geometry_allows_request'] for e in evidence))
        augmented=prompt+'\nUncertain visible-surface measurements follow. They do not establish held state. Judge held state and clearance independently from the images.\n'+json.dumps({'frames':prompt_measurements(evidence)},allow_nan=False)
        event={'tick':tick,'phase':request.get('phase'),'geometry_permits':permits,'evidence_ticks':[x.tick for x in frames]}
        try:raw=self.judge(augmented,frames,request)
        except Exception as exc:
            # Type only: transport messages may carry URLs. Runtime records the same failure.
            self._log({**event,'error_type':type(exc).__name__});raise
        returned=raw
        try:decision=Decision.parse(raw)
        except (ValueError,TypeError):
            # Malformed text passes through unchanged; the controller treats it as no intervention.
            self._log({**event,'raw_model_response':raw,'parse_error':True,'controller_response':raw,'vetoed':False});return raw
        if decision.verdict=='READY_TO_RELEASE' and not permits:
            returned=json.dumps(dict(verdict='WAIT',held=decision.held,central=False,clearance=False,
                views_agree=False,motion='unknown',reason='Perception veto: unvalidated, stale, uncertain or outside measured release envelope.'))
        self._log({**event,'raw_model_response':raw,'parse_error':False,'controller_response':returned,'vetoed':returned!=raw})
        return returned

    def _log(self,event):
        with (self.collector.path/'judge_events.jsonl').open('a') as f:
            f.write(json.dumps(event,allow_nan=False)+'\n')
