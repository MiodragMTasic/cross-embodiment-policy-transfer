"""Bounded Token Factory vision transport. Imported code makes no requests.

A reviewed explicit permit and credential environment variable are required.
Network work runs in a killed-on-timeout subprocess; secrets never enter argv,
request artifacts, exception logs, or this source tree. No retries.
"""
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ENDPOINT = 'https://api.tokenfactory.nebius.com/v1/chat/completions'
MODEL = 'nvidia/Nemotron-3.5-Super-VL'
MAX_OUTPUT = 300


def payload(prompt, frames, request):
    if not 2 <= len(frames) <= 3:
        raise ValueError('requires 2-3 paired timestamps')
    content=[{'type':'text','text':f"Phase: {request['phase']}. Hold since tick {request.get('hold_since_tick')}. Ordered images follow."}]
    for frame in frames:
        for name, png in [('actual exterior',frame.exterior),('actual wrist',frame.wrist)]:
            if not isinstance(png,bytes) or len(png)>1000000 or png[:8]!=b'\x89PNG\r\n\x1a\n' or len(png)<24:
                raise ValueError('invalid bounded PNG')
            if struct.unpack('>II',png[16:24])!=(640,360):
                raise ValueError('expected 640x360')
            content.extend([{'type':'text','text':f'tick {frame.tick}, {name}'},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}}])
    return {'model':MODEL,'messages':[{'role':'system','content':prompt},{'role':'user','content':content}],
            'max_tokens':MAX_OUTPUT,'temperature':0,'response_format':{'type':'json_schema','json_schema':{'name':'release_decision','strict':True,'schema':{'type':'object','additionalProperties':False,'required':['verdict','held','central','clearance','views_agree','motion','reason'],'properties':{'verdict':{'type':'string','enum':['WAIT','READY_TO_RELEASE','UNCERTAIN']},**{k:{'type':'boolean'} for k in ('held','central','clearance','views_agree')},'motion':{'type':'string','enum':['inward','outward','still','unknown']},'reason':{'type':'string','minLength':1,'maxLength':240}}}}},
            'chat_template_kwargs':{'enable_thinking':False}}


class VisionJudge:
    def __init__(self, permit, ledger, sender=None, require_key=True):
        self.permit=json.loads(Path(permit).read_text())
        p=self.permit
        if p.get('review_status')!='OK' or p.get('human_paid_approval') is not True:
            raise PermissionError('review and concrete paid approval required')
        if p.get('model')!=MODEL or p.get('endpoint')!=ENDPOINT:
            raise ValueError('unreviewed endpoint/model')
        if p.get('prices_verified_date')!=time.strftime('%Y-%m-%d',time.gmtime()):
            raise PermissionError('verify current model pricing before launch')
        if p.get('expires_unix',0)<=time.time():
            raise PermissionError('expired permit')
        for key in ('api_cap_usd','reserve_per_call_usd','input_usd_per_million','output_usd_per_million'):
            if not isinstance(p.get(key),(float,int)) or not math.isfinite(p[key]) or p[key]<=0:
                raise ValueError(key)
        if str(Path(ledger).resolve()) != p.get('ledger_path'):
            raise PermissionError('ledger path is not pinned in permit')
        if sender is None and require_key and not os.environ.get('NEBIUS_API_KEY'):
            raise PermissionError('NEBIUS_API_KEY missing before simulator start')
        self.ledger=Path(ledger)
        self.sender=sender or self._send

    def verify_source_lock(self,path):
        p=Path(path)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=self.permit.get('source_lock_sha256'):
            raise PermissionError('source lock not approved')
        for name,expected in json.loads(p.read_text()).items():
            if hashlib.sha256((p.parent/name).read_bytes()).hexdigest()!=expected:
                raise PermissionError('source changed after review')

    def _send(self,body,timeout):
        if not os.environ.get('NEBIUS_API_KEY'):
            raise PermissionError('NEBIUS_API_KEY missing')
        # Child inherits credential environment; captured stderr is never surfaced.
        proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],
            input=json.dumps(body),text=True,capture_output=True,timeout=timeout,check=False)
        if proc.returncode or len(proc.stdout)>131072:
            raise RuntimeError('vision transport failed')
        return json.loads(proc.stdout)

    def __call__(self,prompt,frames,request):
        body=payload(prompt,frames,request)
        p=self.permit
        # Lock spans reservation + request + settlement: shared across episodes.
        self.ledger.parent.mkdir(parents=True,exist_ok=True)
        with self.ledger.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            f.seek(0)
            rows=[json.loads(line) for line in f if line.strip()]
            balance={}
            for r in rows: balance[r['id']]=r['charge_usd']
            reserve=p['reserve_per_call_usd']
            if sum(balance.values())+reserve>p['api_cap_usd']+1e-12:
                raise PermissionError('session API allowance exhausted')
            if time.time()>=p['expires_unix']:
                raise PermissionError('permit expired')
            rid=len(rows)+1
            def append(record):
                f.seek(0,2);f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
            append({'id':rid,'state':'reserved','charge_usd':reserve,'tick':request['tick'],'model':MODEL})
            start=time.monotonic()
            result=self.sender(body,min(10.,request['deadline_s']))
            if time.monotonic()-start>=min(10.,request['deadline_s']):
                raise TimeoutError('late reply')
            if result.get('model')!=MODEL:
                raise ValueError('served model mismatch')
            u=result.get('usage',{})
            pt,ct=u.get('prompt_tokens'),u.get('completion_tokens')
            if any(type(v) is not int or v<0 for v in (pt,ct)):
                raise ValueError('missing usage')
            actual=(pt*p['input_usd_per_million']+ct*p['output_usd_per_million'])/1e6
            append({'id':rid,'state':'settled','charge_usd':actual,'prompt_tokens':pt,'completion_tokens':ct,
                    'model':MODEL,'latency_s':time.monotonic()-start})
            if actual>reserve:
                # Reservation is explicitly an estimate, not a provider hard cap.
                # An underestimated call is recorded and disables further requests.
                append({'id':'reservation-breach','state':'disabled','charge_usd':p['api_cap_usd']})
                raise ValueError('reservation underestimated; transport disabled')
            choice=result['choices'][0]
            if choice.get('finish_reason')!='stop':
                raise ValueError('incomplete reply')
            text=choice['message']['content']
            if not isinstance(text,str) or len(text)>4000:
                raise ValueError('invalid content')
            return text


def worker():
    import urllib.request
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs): raise RuntimeError('redirect refused')
    body=json.loads(sys.stdin.read(3000000))
    req=urllib.request.Request(ENDPOINT,data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+os.environ['NEBIUS_API_KEY'],'Content-Type':'application/json'})
    with urllib.request.build_opener(NoRedirect).open(req,timeout=9) as response:
        data=response.read(131073)
        if len(data)>131072: raise ValueError('response too large')
        sys.stdout.write(data.decode())


if __name__=='__main__':
    try:
        if sys.argv[1:]!=['--worker']: raise ValueError('not a standalone launcher')
        worker()
    except Exception:
        sys.exit(1)
