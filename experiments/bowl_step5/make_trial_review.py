"""Create review-only contact sheets and compact summaries from preserved local runs."""
import json
from pathlib import Path
import subprocess
import sys
import io
from PIL import Image, ImageDraw, ImageFont

case = Path(sys.argv[1])
metric_path, = case.glob('*/strict-metric.json')
d = json.loads(metric_path.read_text())
trace = d.pop('trace')
end = (len(trace)-1)/15
points = {0., min(3.,end), min(6.,end), min(9.,end), min(12.,end), min(18.,end), min(30.,end), end}
for name in ('eligible_open_command', 'measured_open_release', 'invalidated'):
    event = next((e for e in d['release_events'] if e['event'] == name), None)
    if event: points.add(max(0., min(end, event['time_s']+.1)))
if d['stable_success_s']: points.add(max(0., min(end,d['stable_success_s']-.1)))
points = sorted(points)
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',19)
for video in sorted(metric_path.parent.glob('*.mp4')):
    sheet = Image.new('RGB',(1400,286*((len(points)+1)//2)),'white')
    draw = ImageDraw.Draw(sheet)
    for i,t in enumerate(points):
        raw=subprocess.check_output(['/opt/homebrew/bin/ffmpeg','-v','error','-ss',str(t),'-i',str(video),'-frames:v','1','-f','image2pipe','-vcodec','png','pipe:1'])
        image=Image.open(io.BytesIO(raw)).convert('RGB');image.thumbnail((700,260))
        x,y=(i%2)*700,(i//2)*286;sheet.paste(image,(x,y));draw.text((x+8,y+264),f'Video {t:.3f}s',fill='black',font=font)
    label='viewport' if 'viewport' in video.name else 'cameras'
    sheet.save(case/f'review-{label}.jpg',quality=92)
d['trace_rows']=len(trace)
d['raw_predicate_true_rows']=sum(r['upstream_predicate'] for r in trace)
d['review_warning']='Video timestamps label frames; exact pre/post action/event times are in original metric/capture.'
(case/'review-summary.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'case':case.name,'schema':d['schema'],'rows':len(trace),'automated_pass':d['stable_success_s'] is not None,'events':d['release_events']}))
