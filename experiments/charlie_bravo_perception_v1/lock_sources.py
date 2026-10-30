"""Create an integrity lock after local changes; this does NOT grant launch approval."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
files=list(P.glob('*.py'))
files += [P.parent/'charlie_bravo_v1'/n for n in ('runner_B.py','controller.py','runtime.py','transport.py','vision_images.py','vision_prompt.txt')]
files += [P.parent/'bowl_step8/step8-source-lock.sha256']
import os
lock={os.path.relpath(f,P):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
(P/'source-lock.json').write_text(json.dumps(lock,indent=2)+'\n')
print('Locked',len(lock),'files. Review and a current launch permit are still required.')
