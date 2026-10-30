#!/bin/bash
# Intent demo v1 clips, on the Mac, from evidence already copied back and collected (DEMO_PLAN.md, launch steps 9-11).
#   make_clips.sh runs/<ts>     cards -> one clip per episode and per request not run -> summary -> reel -> MANIFEST.json
#   make_clips.sh --mockup      mockup/: hand-written specs on the step 8 before-phase A/P0/4004 run, labelled MOCKUP
# Clips are 1920x1080 H.264, made with ffmpeg from the runs' own saved videos; text is drawn by render_clip.py (Pillow,
# SF Pro / SF Mono from /System/Library/Fonts) because this ffmpeg build has no drawtext. Never overwrites: a session
# gets its clips once.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py="$here/.venv/bin/python"
[ -x "$py" ] && "$py" -c 'import PIL, h5py, numpy' 2>/dev/null || {
 echo "needs $here/.venv with pillow, h5py, numpy (DEMO_PLAN.md, launch step 0)"; exit 2; }
render() { "$py" -B "$here/render_clip.py" "$@"; }

if [ "${1:-}" = --mockup ]; then
 out="$here/mockup"
 "$py" -B "$here/demo_session.py" mockup "$out" > /dev/null
 render "$out/MOCKUP_put_the_fruit_away.json" "$out/MOCKUP_put_the_fruit_away.mp4"
 render "$out/MOCKUP_pick_up_the_bowl_and_put_it_on_the_banana.json" "$out/MOCKUP_pick_up_the_bowl_and_put_it_on_the_banana.mp4"
 # the left-of mockup is stills only: its footage is a run told to put the banana IN the bowl, so a moving clip of it
 # could be mistaken for a left-of attempt
 render "$out/MOCKUP_put_the_banana_beside_the_bowl_on_its_left.json" "$out/MOCKUP_put_the_banana_beside_the_bowl_on_its_left_spec.png" --still 4.5
 render "$out/MOCKUP_put_the_banana_beside_the_bowl_on_its_left.json" "$out/MOCKUP_put_the_banana_beside_the_bowl_on_its_left_verdict.png" --still 99
 render "$out/MOCKUP_summary.json" "$out/MOCKUP_summary.png" --still 0
 for f in "$out"/MOCKUP_*.mp4; do render "${f%.mp4}.json" "${f%.mp4}_verdict.png" --still 99; done
 (cd "$out" && ls MOCKUP_* | grep -v SHA256SUMS | tr "\n" "\0" | xargs -0 shasum -a 256 > MOCKUP_SHA256SUMS.tmp && mv MOCKUP_SHA256SUMS.tmp MOCKUP_SHA256SUMS)
 ls -la "$out"
 exit 0
fi

[ $# = 1 ] || { echo 'usage: make_clips.sh runs/<ts> | --mockup'; exit 2; }
s=$(cd "$1" && pwd)
[ -f "$s/episodes/summary.json" ] || { echo "REFUSE: $s/episodes/summary.json missing; run demo_session.py collect first"; exit 3; }
[ ! -e "$s/clips" ] || { echo "REFUSE: $s/clips exists; clips are made once per session"; exit 3; }
names=$("$py" -B "$here/demo_session.py" cards "$s")   # macOS /bin/bash is 3.2: no mapfile
[ -n "$names" ]
mkdir "$s/clips"
: > "$s/clips/reel.txt"
for name in $names summary; do   # card names are [a-z0-9_] slugs: word splitting is safe
 render "$s/cards/$name.json" "$s/clips/$name.mp4"
 render "$s/cards/$name.json" "$s/clips/$name.png" --still 99 > /dev/null
 printf "file '%s'\n" "$name.mp4" >> "$s/clips/reel.txt"
done
# every clip, in the declared order, then the closing summary: the whole session in one file
(cd "$s/clips" && ffmpeg -v error -f concat -safe 0 -i reel.txt -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
 -movflags +faststart reel.mp4)
"$py" -B "$here/demo_session.py" manifest "$s"
ls -la "$s/clips"
