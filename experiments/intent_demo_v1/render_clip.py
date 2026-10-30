"""Render one demo clip (1920x1080, 30 fps, H.264) from a card JSON written by demo_session.py cards (or mockup).

ffmpeg decodes the run's own saved video (the exterior-camera half of RoboLab's sensor video, the view GR00T gets) and
encodes the result; Pillow draws the text, because this ffmpeg build has no drawtext or libass. Nothing is re-rendered in
simulation. Timeline: the request types out in the top band; Nemotron's spec appears in the left panel, key fields large
and the raw JSON under them; the run plays on the right (sped up, and labelled so, when it is long); the checker's verdict
replaces the left panel while the run's last frame stays on the right. A request that is not run shows the start scene,
greyed, with the refusal; an episode that started but saved no video shows a "no footage" panel. The verdict lists the
spec's own check and the demo floor, each with its items, then lines shown but not graded. A summary card lists every
episode in the declared order.

  render_clip.py <card.json> <out.mp4>            render
  render_clip.py <card.json> <out.png> --still T  one frame at T seconds (T past the end gives the final frame)
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

W, H, FPS = 1920, 1080, 30
SRC_FPS = 15
SANS, MONO = '/System/Library/Fonts/SFNS.ttf', '/System/Library/Fonts/SFNSMono.ttf'
BG, BAND, RULE = (11, 13, 16), (18, 21, 25), (40, 44, 50)
TEXT, MUTED, DIM = (236, 238, 240), (152, 158, 165), (98, 104, 112)
GREEN, RED, AMBER = (112, 204, 124), (240, 116, 104), (246, 196, 82)
TONE = {'ok': GREEN, 'warn': AMBER, 'bad': RED, None: TEXT}
M = 72                       # side margin
BAND_H = 160
LEFT_W = 760                 # left panel: x M .. M + LEFT_W
VID_X, VID_Y, VID_W, VID_H = 888, 214, 960, 540
TYPE_CPS = 22                # typing speed, characters per second
ROW_GAP_S, FADE_S = .35, .25
END_S, END_FADE_S, HOLD_S = 3.5, .35, .4
NOT_RUN_S = 2.6
MAX_ROBOT_S = 10.0
SPEEDS = (1, 1.5, 2, 3, 4, 5, 6, 8)
STRIP_H = 46


def font(size, weight='Regular', mono=False):
    f = ImageFont.truetype(MONO if mono else SANS, size)
    f.set_variation_by_name(weight)
    return f


def wrap(text, f, width):
    lines, line = [], ''
    for word in str(text).split():
        trial = f'{line} {word}'.strip()
        if f.getlength(trial) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def ease(x):
    x = min(max(x, 0.), 1.)
    return 1 - (1 - x) ** 3


def layer(size):
    return Image.new('RGBA', size, (0, 0, 0, 0))


def paste_alpha(base, img, xy, alpha):
    if alpha <= 0:
        return
    if alpha < 1:
        img = img.copy()
        img.putalpha(img.getchannel('A').point(lambda v: int(v * alpha)))
    base.alpha_composite(img, xy)


def icon(kind, d=104):
    """A drawn check, cross or dash in a ring (no font glyph needed)."""
    s = 4
    im = layer((d * s, d * s))
    g = ImageDraw.Draw(im)
    color = {'ok': GREEN, 'bad': RED, 'none': AMBER}[kind]
    w = max(3, round(d / 15)) * s
    g.ellipse([w // 2, w // 2, d * s - w // 2, d * s - w // 2], outline=color, width=w)
    c, u = d * s / 2, d * s / 104
    if kind == 'ok':
        g.line([(c - 22 * u, c + 1 * u), (c - 6 * u, c + 17 * u), (c + 24 * u, c - 15 * u)], fill=color, width=w, joint='curve')
    elif kind == 'bad':
        r = 17 * u
        g.line([(c - r, c - r), (c + r, c + r)], fill=color, width=w)
        g.line([(c - r, c + r), (c + r, c - r)], fill=color, width=w)
    else:
        g.line([(c - 20 * u, c), (c + 20 * u, c)], fill=color, width=w)
    return im.resize((d, d), Image.LANCZOS)


def probe_frames(video):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
                          'stream=nb_read_frames', '-of', 'csv=p=0', str(video)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


class Frames:
    """Sequential frames of the cropped, sped-up, resampled source, already at panel size."""

    def __init__(self, video, crop, speed):
        cw, ch, cx, cy = crop
        vf = f'crop={cw}:{ch}:{cx}:{cy},setpts=PTS/{speed},fps={FPS},scale={VID_W}:{VID_H}:flags=lanczos'
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(video), '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)  # an early close is a broken pipe, not an error
        self.last = None

    def next(self):
        raw = self.p.stdout.read(VID_W * VID_H * 3)
        if len(raw) == VID_W * VID_H * 3:
            self.last = Image.frombytes('RGB', (VID_W, VID_H), raw)
        return self.last

    def close(self):
        self.p.stdout.close()
        self.p.wait()


def chrome(c, base):
    """Band, DEMO/MOCKUP pill, mockup strip: on every card."""
    g = ImageDraw.Draw(base)
    g.rectangle([0, 0, W, BAND_H], fill=BAND + (255,))
    g.line([(0, BAND_H), (W, BAND_H)], fill=RULE, width=2)
    pill = 'MOCKUP' if c.get('mockup') else 'DEMO'
    pf = font(24, 'Bold')
    pw = int(pf.getlength(pill)) + 36
    g.rounded_rectangle([W - M - pw, 44, W - M, 84], radius=20, outline=AMBER, width=2)
    g.text((W - M - pw / 2, 64), pill, font=pf, fill=AMBER, anchor='mm')
    g.text((W - M, 106), 'separate from the step 8 experiment', font=font(22), fill=MUTED, anchor='ra')
    if c.get('mockup'):
        g.rectangle([0, H - STRIP_H, W, H], fill=AMBER + (255,))
        g.text((W / 2, H - STRIP_H / 2), c['mockup_note'], font=font(21, 'Semibold'), fill=(20, 18, 12), anchor='mm')


class Clip:
    def __init__(self, card):
        self.c = card
        self.footage = card.get('video') is not None
        self.run = card['kind'] == 'run' and self.footage   # plays the run's video
        self.bottom = H - (STRIP_H if card.get('mockup') else 0)
        src_s = probe_frames(card['video']) / SRC_FPS if self.run else 0
        self.speed = next((k for k in SPEEDS if src_s / k <= MAX_ROBOT_S), SPEEDS[-1]) if self.run else 1
        self.robot_s = src_s / self.speed if self.run else NOT_RUN_S
        self.t_typed = .3 + len(card['instruction']) / TYPE_CPS
        self.t_spec = self.t_typed + .3
        self.t_robot = self.t_spec + len(card['spec_rows']) * ROW_GAP_S + .4
        self.t_end = self.t_robot + self.robot_s + HOLD_S
        self.duration = self.t_end + END_S
        self.frames = Frames(card['video'], card['video_crop'], self.speed) if self.footage else None
        self.first = self.frames.next() if self.footage else Image.new('RGB', (VID_W, VID_H), BAND)
        self.shown = 0
        self.build_static()

    def build_static(self):
        c = self.c
        base = Image.new('RGBA', (W, H), BG + (255,))
        chrome(c, base)
        g = ImageDraw.Draw(base)
        g.text((M, 214), c['nemotron_model'], font=font(24, 'Medium', mono=True), fill=MUTED, anchor='la')
        g.text((M, 250), 'Nemotron on Nebius Token Factory → task spec', font=font(22), fill=DIM, anchor='la')
        g.rectangle([VID_X - 2, VID_Y - 2, VID_X + VID_W + 1, VID_Y + VID_H + 1], outline=RULE, width=2)
        g.text((VID_X, VID_Y + VID_H + 30), c['robot_caption'], font=font(24), fill=MUTED, anchor='la')
        if self.run and self.speed > 1:
            g.text((VID_X + VID_W, VID_Y + VID_H + 30), f'{self.speed:g}× speed', font=font(24, 'Semibold'), fill=TEXT, anchor='ra')
        for i, line in enumerate(wrap(c.get('run_label', ''), font(20, mono=True), VID_W)[:2]):
            g.text((VID_X, VID_Y + VID_H + 72 + 28 * i), line, font=font(20, mono=True), fill=DIM, anchor='la')
        self.base = base
        self.rows = [self.row_image(r) for r in c['spec_rows']]
        self.json_lines = c['raw_json'].splitlines()
        self.json_font = font(17, mono=True)
        self.type_font = font(56, 'Semibold')
        if not self.footage:
            still = self.first.convert('RGBA')
            g2 = ImageDraw.Draw(still)
            g2.text((VID_W / 2, VID_H / 2 - 50), 'NO FOOTAGE', font=font(40, 'Bold'), fill=AMBER, anchor='mm')
            for i, line in enumerate(wrap(c.get('no_footage_reason', ''), font(25), VID_W - 160)[:3]):
                g2.text((VID_W / 2, VID_H / 2 + 14 + 36 * i), line, font=font(25), fill=TEXT, anchor='mm')
            self.still = still.convert('RGB')
        elif not self.run:
            still = ImageEnhance.Color(self.first).enhance(.15)
            still = ImageEnhance.Brightness(still).enhance(.42).convert('RGBA')
            g2 = ImageDraw.Draw(still)
            g2.rounded_rectangle([VID_W / 2 - 220, VID_H / 2 - 100, VID_W / 2 + 220, VID_H / 2 - 30], radius=12, outline=AMBER, width=3)
            g2.text((VID_W / 2, VID_H / 2 - 65), 'ROBOT NOT RUN', font=font(40, 'Bold'), fill=AMBER, anchor='mm')
            for i, line in enumerate(wrap(c['not_run_reason'], font(25), VID_W - 160)[:3]):
                g2.text((VID_W / 2, VID_H / 2 + 14 + 36 * i), line, font=font(25), fill=TEXT, anchor='mm')
            self.still = still.convert('RGB')
        self.endcard = self.end_image()

    def row_image(self, cells):
        """One spec row: a single wide field, or several short fields side by side."""
        kf = font(22, mono=True)
        width = LEFT_W // len(cells)
        blocks = []
        for key, value, tone in cells:
            if key == 'robot instruction':
                vf, text = font(32, 'Medium'), f'“{value}”'
            elif key in ('reason', 'source', 'note'):
                vf, text = font(25), value
            else:
                vf, text = font(44 if len(cells) > 1 else 36, 'Semibold'), value
            blocks.append((key, wrap(text, vf, width - 24)[:4], vf, TONE[tone]))
        h = max(32 + int(vf.size * 1.22) * len(lines) for _, lines, vf, _ in blocks) + 20
        im = layer((LEFT_W, h))
        g = ImageDraw.Draw(im)
        for i, (key, lines, vf, color) in enumerate(blocks):
            x = i * width
            g.text((x, 0), key, font=kf, fill=MUTED + (255,))
            for j, line in enumerate(lines):
                g.text((x, 32 + j * int(vf.size * 1.22)), line, font=vf, fill=color + (255,))
        return im

    def end_image(self):
        c, v = self.c, self.c['verdict']
        im = layer((W, H))
        g = ImageDraw.Draw(im)
        g.rectangle([0, BAND_H + 2, VID_X - 6, self.bottom], fill=BG + (255,))
        kind = {True: 'ok', False: 'bad', None: 'none'}[v['ok']]
        tone = {True: GREEN, False: RED, None: AMBER}[v['ok']]
        items = v.get('checks', [])
        dense = len(items) > 6
        im.alpha_composite(icon(kind, 80 if dense else 96), (M, 200 if dense else 230))
        y = 300 if dense else 360
        big = font(48 if dense else 54, 'Semibold')
        for i, line in enumerate(v['lines']):
            color = tone if i == len(v['lines']) - 1 else TEXT
            for part in wrap(line, big, LEFT_W):
                g.text((M, y), part, font=big, fill=color + (255,))
                y += 60 if dense else 68
        y += 10 if dense else 16
        df = font(22 if dense else 26)
        for line in wrap(v['detail'], df, LEFT_W)[:3 if dense else 6]:
            g.text((M, y), line, font=df, fill=MUTED + (255,))
            y += 30 if dense else 38
        y += 12 if dense else 22
        for item in items:
            label, ok = item[0], item[1]
            mark = {True: 'ok', False: 'bad', None: 'none'}[ok]
            if len(item) > 2 and item[2] == 'head':   # a section: the spec's check, the floor, or shown-only lines
                y += 4
                if ok is not None or label != 'Shown, not graded':
                    im.alpha_composite(icon(mark, 30), (M, y))
                g.text((M + 44, y + 15), label, font=font(24, 'Semibold'), fill=TEXT + (255,), anchor='lm')
                y += 36
                continue
            im.alpha_composite(icon(mark, 24), (M + 44, y + 2) if dense else (M, y))
            lf = font(22 if dense else 26)
            text = label
            while lf.getlength(text) > LEFT_W - 84 and len(text) > 4:
                text = text[:-2] + '\u2026'
            g.text((M + 80, y + 14) if dense else (M + 52, y + 17), text, font=lf,
                   fill=(MUTED if ok is None else TEXT) + (255,), anchor='lm')
            y += 30 if dense else 48
        g.rectangle([VID_X - 4, VID_Y - 4, VID_X + VID_W + 3, VID_Y + VID_H + 3], outline=tone + (255,), width=4)
        fy = self.bottom - 28 - 30 * len(c['footer'])
        g.rectangle([0, fy - 20, W, self.bottom], fill=BG + (255,))
        for line in c['footer']:
            g.text((M, fy), line, font=font(20, mono=True), fill=DIM + (255,))
            fy += 30
        return im

    def robot_frame(self, t):
        if not self.run:
            return self.still
        if t < self.t_robot:
            return ImageEnhance.Brightness(self.first).enhance(.55)
        want = min(int((t - self.t_robot) * FPS), int(self.robot_s * FPS))
        while self.shown < want:
            self.frames.next()
            self.shown += 1
        return self.frames.last

    def frame(self, t):
        c = self.c
        im = self.base.copy()
        g = ImageDraw.Draw(im)
        n = max(0, min(len(c['instruction']), int((t - .3) * TYPE_CPS)))
        typed = f'“{c["instruction"][:n]}' + ('”' if n == len(c['instruction']) else '')
        g.text((M, BAND_H / 2), typed, font=self.type_font, fill=TEXT, anchor='lm')
        if t < self.t_typed + .9 and int(t * 2.5) % 2 == 0:
            x = M + self.type_font.getlength(typed) + 6
            g.rectangle([x, BAND_H / 2 - 30, x + 4, BAND_H / 2 + 30], fill=TEXT)
        y = 310
        for i, r in enumerate(self.rows):
            a = ease((t - self.t_spec - i * ROW_GAP_S) / FADE_S)
            paste_alpha(im, r, (M, y + int(10 * (1 - a))), a)
            y += r.height + 6
        jy = y + 20
        g.line([(M, jy - 10), (M + LEFT_W, jy - 10)], fill=RULE, width=1)
        shown = int(max(0., t - self.t_spec) * 40)
        for line in self.json_lines[:min(shown, (self.bottom - 24 - jy) // 22)]:
            while self.json_font.getlength(line) > LEFT_W:
                line = line[:-2] + '\u2026'
            g.text((M, jy), line, font=self.json_font, fill=DIM)
            jy += 22
        im.paste(self.robot_frame(t), (VID_X, VID_Y))
        if self.run:
            p = min(max((t - self.t_robot) / self.robot_s, 0.), 1.)
            g.rectangle([VID_X, VID_Y + VID_H + 4, VID_X + int(VID_W * p), VID_Y + VID_H + 7], fill=TEXT)
        paste_alpha(im, self.endcard, (0, 0), ease((t - self.t_end) / END_FADE_S))
        return im.convert('RGB')

    def frames_out(self):
        for k in range(int(math.ceil(self.duration * FPS))):
            yield self.frame(k / FPS)

    def close(self):
        if self.frames:
            self.frames.close()


class Summary:
    """Every episode in the declared order, with its verdict. A still, held for 6 s in the reel."""

    def __init__(self, card):
        self.c, self.duration = card, 6.0
        im = Image.new('RGBA', (W, H), BG + (255,))
        chrome(card, im)
        g = ImageDraw.Draw(im)
        g.text((M, BAND_H / 2), card['title'], font=font(56, 'Semibold'), fill=TEXT, anchor='lm')
        g.text((M, 210), card['subtitle'], font=font(24), fill=MUTED, anchor='la')
        y = 280
        rh = min(118, (H - (STRIP_H if card.get('mockup') else 0) - 40 - y) // max(1, len(card['rows'])))
        for r in card['rows']:
            g.text((M, y + 20), str(r['order']), font=font(30, 'Medium', mono=True), fill=DIM, anchor='la')
            im.alpha_composite(icon(r['mark'], 56), (M + 56, y + 6))
            g.text((M + 140, y + 4), f"“{r['request']}”", font=font(34, 'Semibold'), fill=TEXT, anchor='la')
            g.text((M + 140, y + 52), f"{r['source']} · layout {r['layout']}", font=font(22), fill=MUTED, anchor='la')
            color = {'ok': GREEN, 'bad': RED, 'none': AMBER}[r['mark']]
            for i, line in enumerate(wrap(r['text'], font(26), 600)[:2]):
                g.text((W - M, y + 8 + 34 * i), line, font=font(26), fill=color, anchor='ra')
            g.line([(M, y + rh - 12), (W - M, y + rh - 12)], fill=RULE, width=1)
            y += rh
        self.img = im.convert('RGB')

    def frame(self, t):
        return self.img

    def frames_out(self):
        for _ in range(int(self.duration * FPS)):
            yield self.img

    def close(self):
        pass


def make(card):
    return Summary(card) if card['kind'] == 'summary' else Clip(card)


def render(clip, out):
    enc = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                            '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
    try:
        for f in clip.frames_out():
            enc.stdin.write(f.tobytes())
    finally:
        enc.stdin.close()
        clip.close()
    if enc.wait():
        raise RuntimeError('ffmpeg encode failed')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('card')
    ap.add_argument('out')
    ap.add_argument('--still', type=float)
    a = ap.parse_args(argv)
    clip = make(json.loads(Path(a.card).read_text()))
    if a.still is not None:
        img = clip.frame(0)
        if isinstance(clip, Clip):   # step through every frame up to T so the decoder position matches the clip
            for k in range(int(min(a.still, clip.duration - 1 / FPS) * FPS) + 1):
                img = clip.frame(k / FPS)
        img.save(a.out)
        clip.close()
    else:
        render(clip, a.out)
    print(json.dumps({'out': a.out, 'seconds': round(clip.duration, 2), 'speed': getattr(clip, 'speed', None)}))


if __name__ == '__main__':
    sys.exit(main())
