"""Authorized saved-video screening only. Never starts or connects to a VM."""
from pathlib import Path
import base64,hashlib,io,json,os,subprocess,sys,time,urllib.request
import numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
ROOT=PACKAGE.parents[1]
sys.path.insert(0,str(PACKAGE))
from controller import Frame,Decision
from transport import payload,MODEL,ENDPOINT
RUNS=ROOT/'assets/report/evidence/2026-10-25/step8_after_v1/runs'
CASES=[(4003,750,False),(4003,900,False),(4003,1200,True),(4003,1500,True),(4003,1870,False),(4003,2000,False),(4009,1500,False)]
CAP=.05
RESERVE=.02 # conservative ~65k input tokens plus 300 output; no retries

def prepare():
    jobs=[]
    for layout,tick,positive in CASES:
        run=RUNS/f'step8_B_P1_l{layout}_s0_v1'
        trace=[json.loads(x) for x in (run/'gate_trace.jsonl').read_text().splitlines()]
        obj=np.load(run/'tracking.npz')['objects_banana_bowl']
        ticks=[(tick-d)//3*3 for d in (12,6,0)]
        labels=[]
        for t in ticks:
            r=trace[t]
            labels.append({'video_frame':t//3,'video_s':t/15,'control_index':t,'trace_step':r['step'] if 'step' in r else None,'contact':r['post']['contact'],'command':r['command'],'xy_distance_m':float(np.linalg.norm(obj[t,0,:2]-obj[t,1,:2]))})
        case=HERE/f'l{layout}_t{tick}';case.mkdir(exist_ok=True)
        (case/'labels_not_sent.json').write_text(json.dumps({'expected_ready':positive,'trace':labels},indent=2))
        originals=[]
        for t in ticks:
            b=subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-ss',str(t/15),'-i',str(run/'views.mp4'),'-frames:v','1','-f','image2pipe','-vcodec','png','-'],capture_output=True,check=True).stdout
            im=Image.open(io.BytesIO(b)).convert('RGB');assert im.size==(1920,360)
            pair=[]
            for x,name in [(0,'exterior'),(1280,'wrist')]:
                view=im.crop((x,0,x+640,360));ImageDraw.Draw(view).rectangle((0,0,639,30),fill='black')
                view.save(case/f'{t}_{name}_640.png');pair.append(view)
            originals.append((t,pair))
        for width,height in [(320,180),(640,360)]:
            folder=case/f'{width}x{height}';folder.mkdir(exist_ok=True)
            frames=[];manifest=[]
            for t,pair in originals:
                buffers=[]
                for name,im in zip(('exterior','wrist'),pair):
                    b=io.BytesIO();im.resize((width,height),Image.Resampling.LANCZOS).save(b,format='PNG');data=b.getvalue()
                    fp=folder/f'{t}_{name}.png';fp.write_bytes(data);buffers.append(data)
                    manifest.append({'file':fp.name,'sha256':hashlib.sha256(data).hexdigest()})
                frames.append(Frame(t,*buffers))
            # Build exact production schema/prompt with bounded small images, then
            # substitute only higher-resolution image bytes for the comparison.
            small=[]
            for t,pair in originals:
                bs=[]
                for im in pair:
                    b=io.BytesIO();im.resize((320,180)).save(b,format='PNG');bs.append(b.getvalue())
                small.append(Frame(t,*bs))
            body=payload((PACKAGE/'vision_prompt.txt').read_text(),small,{'phase':'POLICY','tick':ticks[-1],'hold_since_tick':None})
            parts=[part for part in body['messages'][1]['content'] if part['type']=='image_url']
            for part,data in zip(parts,[b for f in frames for b in (f.exterior,f.wrist)]):
                assert len(data)<1000000
                part['image_url']['url']='data:image/png;base64,'+base64.b64encode(data).decode()
            (folder/'request.json').write_text(json.dumps(body))
            (folder/'image_manifest.json').write_text(json.dumps(manifest,indent=2))
            jobs.append({'folder':str(folder.relative_to(HERE)),'positive':positive,'resolution':width})
    (HERE/'jobs.json').write_text(json.dumps(jobs,indent=2))
    print('Prepared',len(jobs),'requests; no calls made')

def run():
    jobs=json.loads((HERE/'jobs.json').read_text());ledger=HERE/'spend.jsonl'
    if ledger.exists():raise RuntimeError('Existing ledger: do not automatically rerun')
    cat=json.loads((HERE/'current_catalog.json').read_text())['entries'][0]['flavors'][0]
    pin=cat['input_price_per_million_tokens'];pout=cat['output_price_per_million_tokens']
    assert (pin,pout)==(.3,.9)
    key=(Path.home()/'.config/nebius/token-factory.key').read_text().strip()
    accounted=0.;results=[]
    for job in jobs:
        if accounted+RESERVE>CAP+1e-12:break
        folder=HERE/job['folder'];body=json.loads((folder/'request.json').read_text())
        def record(x):
            with ledger.open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
        record({'job':job['folder'],'state':'reserved','charge':RESERVE});accounted+=RESERVE
        start=time.monotonic();row={**job,'ready':False,'valid':False}
        try:
            env=os.environ.copy();env['NEBIUS_API_KEY']=key
            proc=subprocess.run([sys.executable,str(PACKAGE/'transport.py'),'--worker'],input=json.dumps(body),text=True,capture_output=True,env=env,timeout=10)
            if proc.returncode:raise RuntimeError('worker_failed')
            raw=json.loads(proc.stdout);(folder/'response.json').write_text(json.dumps(raw,indent=2))
            u=raw['usage'];pt=u['prompt_tokens'];ct=u['completion_tokens'];assert type(pt)==int and type(ct)==int and pt>=0 and ct>=0
            actual=(pt*pin+ct*pout)/1e6
            record({'job':job['folder'],'state':'settled','charge':actual,'prompt_tokens':pt,'completion_tokens':ct})
            accounted+=actual-RESERVE
            row.update(cost=actual,prompt_tokens=pt,completion_tokens=ct)
            if actual>RESERVE:raise RuntimeError('reservation_breach')
            assert raw['model']==MODEL
            ch=raw['choices'][0];assert ch['finish_reason']=='stop'
            d=Decision.parse(ch['message']['content']);row.update(valid=True,ready=d.ready,decision=d.__dict__)
        except Exception as e:row['error']=type(e).__name__
        row['latency_s']=time.monotonic()-start;results.append(row)
        (HERE/'results.json').write_text(json.dumps({'accounted_usd':accounted,'cap_usd':CAP,'results':results},indent=2))
        print(job['folder'], 'READY' if row['ready'] else 'not READY',row.get('error',''),flush=True)
        if row.get('error'):break # no retries; preserve bounded unknown cost
    print('Accounted USD',accounted)

if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif sys.argv[1:]==['run']:run()
    else:raise SystemExit('prepare or run')
