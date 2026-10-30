"""Intent demo v1, Mac side: one launch = one session folder runs/<UTC ts>/ holding the proof. DEMO, separate from step 8.

  runs/<ts>/DEMO_LABEL.txt
  runs/<ts>/declared_plan.json        the run order declared before any call or episode (sha256 pinned below)
  runs/<ts>/snapshot/                 the prompt and the demo code as they were when the session opened
  runs/<ts>/nemotron/<slug>/          per request: input.txt, every HTTP request/response body, r{n}.json, agreement.json
  runs/<ts>/specs/                    the robot's specs (Nemotron repeat 1 when all repeats agree) and the literal controls
  runs/<ts>/vm_plan_stop{N}.json      what run_intent_demo.sh runs; plan_note_stop{N}.json says why anything is left out
  runs/<ts>/vm/                       copy-back from the VM, checked file by file against vm/SHA256SUMS made on the VM
  runs/<ts>/episodes/<n>_<tag>/       per declared episode: result.json (instruction received, checker output, files)
  runs/<ts>/cards/, clips/            make_clips.sh
  runs/<ts>/MANIFEST.json             every file's sha256, and what ties to what

  demo_session.py new                              open a session, print its path (refused once any session asked Nemotron)
  demo_session.py compile <session>                3 requests x 3 Nemotron calls (+1 retry each at most, so up to 18):
                                                   spends money, Miodrag's yes first; refused if any ledger already holds
                                                   an intent_demo_v1 row (Nemotron is asked once per demo, never again)
  demo_session.py plan <session> [--stop 2|4|6]    the VM plan for the first N declared episodes (default 4)
  demo_session.py collect <session>                after copy-back: verify, read the instruction GR00T got, run the checker
  demo_session.py cards <session>                  one card per episode and per request not run (input of render_clip.py)
  demo_session.py manifest <session>
  demo_session.py mockup <out_dir>                 MOCKUP cards: hand-written specs on step 8 before-phase footage
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import checker  # noqa: E402
import demo_plan  # noqa: E402
import intent_to_spec as its  # noqa: E402

DECLARED = HERE / 'declared_plan.json'
DECLARED_SHA256 = 'dac11c7062c18abbb4ceb42a9c9772290bfa6ca96d74c58a50a115764de11581'
LITERAL_SHA256 = {'literal_put_the_fruit_away.json': '2144a3aaa70934c5337251480ee3ec8c6a632bd7900ce83ce2cfea9173478fc0',
                  'literal_put_the_banana_beside_the_bowl_on_its_left.json':
                      'be62e0584bfbf049e4d5b57d799d5252ea6e48c1e178b3ce99ee0d3a50bd5571'}
REPEATS = 3
STOPS = (2, 4, 6)
DEFAULT_STOP = 4   # the team lead's go/no-go default
TASK = 'BananaInBowlUniformInitPose10cmTask'
SUPERVISOR = 'runs/intentdemo_v1-supervisor'
CODE = ['intent_to_spec.py', 'checker.py', 'demo_session.py', 'render_clip.py', 'make_clips.sh', 'demo_plan.py',
        'a_instruction_wrapper.py', 'run_intent_demo.sh', 'intent-demo-lock.sha256']
EVIDENCE = HERE.parents[1] / 'assets/report/evidence/2026-10-24/step8_before_v1'
BEFORE_4004 = EVIDENCE / 'RoboLab/output/step8_A_P0_l4004_s0_v1' / TASK
EXTERIOR_CROP = [640, 360, 0, 0]   # left half of RoboLab's 1280x360 sensor video: the exterior camera GR00T gets
GROOT = 'GR00T N1.7 DROID, P0 snapshot 05e7cc97, on a Nebius GPU VM, Isaac Sim 5.1, RoboLab b6cfcf6'
LABEL = ('DEMO. Intent demo v1: Nemotron (Token Factory) writes a checkable task, GR00T (VM) attempts it, the checker grades '
         'it against Nemotron\'s own spec. Separate from the step 8 experiment: never scored or pooled with it.')
REL_WORD = {'in': 'IN', 'on': 'ON', 'left_of': 'LEFT OF', 'right_of': 'RIGHT OF', 'beside': 'BESIDE'}
REL_PHRASE = {'in': 'in the', 'on': 'on the', 'left_of': 'left of the', 'right_of': 'right of the', 'beside': 'beside the'}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def session_dir(path):
    s = Path(path).resolve()
    if not s.is_relative_to(its.RUNS.resolve()) or not (s / 'declared_plan.json').is_file():
        raise SystemExit(f'{path} is not a session made by demo_session.py new')
    if sha_file(s / 'declared_plan.json') != DECLARED_SHA256:
        raise SystemExit(f'{s}/declared_plan.json differs from the declared plan')
    return s


def copy_in(src, dst, want=None):
    got = sha_file(src)
    if want and got != want:
        raise SystemExit(f'{src} sha256 {got} != pinned {want}')
    shutil.copyfile(src, its.guarded(dst))
    return got


def asked_sessions():
    """Sessions that asked Nemotron with the current prompt version (their snapshot holds this prompt file)."""
    return sorted(p.parent.name for p in its.RUNS.glob('*/nemotron') if (p.parent / 'snapshot' / its.PROMPT_FILE.name).is_file())


def demo_rows():
    """Rows of the current prompt version (its.PURPOSE) in any ledger: the demo's, the model ladder's and the sealed one.
    Earlier versions' rows do not block a new version, but still count toward the demo cap (its.demo_spent)."""
    return [r for path in (its.ledger(), its.LADDER_LEDGER, its.SEALED_LEDGER) for r in its.tf.ledger_rows(path)
            if str(r.get('purpose', '')).startswith(its.PURPOSE)]


def new():
    if sha_file(DECLARED) != DECLARED_SHA256:
        raise SystemExit('declared_plan.json changed after it was declared; refusing')
    if asked_sessions():
        raise SystemExit(f'refusing: session(s) {asked_sessions()} already asked Nemotron with this prompt version; it is asked once per version')
    s = its.RUNS / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    its.guarded(s / 'DEMO_LABEL.txt').write_text(LABEL + '\n')
    copy_in(DECLARED, s / 'declared_plan.json', DECLARED_SHA256)
    copy_in(its.PROMPT_FILE, s / 'snapshot' / its.PROMPT_FILE.name, its.PROMPT_SHA256)
    for name in CODE:
        copy_in(HERE / name, s / 'snapshot' / name)
    return s


def compile_requests(session, client=None):
    """Every declared request, in order, REPEATS calls each. A request whose repeats disagree gets no robot spec.
    Refused before any call if any ledger already holds a row of this prompt version, or another session asked with it."""
    if demo_rows():
        raise SystemExit(f'refusing: the ledgers already hold {len(demo_rows())} {its.PURPOSE} row(s); Nemotron is asked once per prompt version')
    others = [s for s in asked_sessions() if s != session.name]
    if others:
        raise SystemExit(f'refusing: session(s) {others} already asked Nemotron')
    client = client or its.DemoClient()
    out = []
    for r in read(session / 'declared_plan.json')['requests']:
        ok, summary = its.run_live(r['text'], session, REPEATS, client=client)
        out.append(summary)
    its.write_json(session / 'nemotron' / 'summary.json', out)
    return out


def spec_for(session, e):
    """(spec path or None, reason it cannot run or None) for one declared episode."""
    slug = its.slug(e['request'])
    dst = session / 'specs' / f'{slug}.json'
    if e['source'] == 'literal':
        if not dst.exists():   # P1: the raw words are graded with this session's Nemotron check, so it must exist
            return None, 'raw-words control: no agreed Nemotron check for this request in this session to grade it with'
        name = f'literal_{slug}.json'
        lit = session / 'specs' / name
        if not lit.exists():
            copy_in(HERE / 'specs' / name, lit, LITERAL_SHA256[name])
        return lit, None
    if not dst.exists():
        agree = session / 'nemotron' / slug / 'agreement.json'
        if not agree.exists():
            return None, 'Nemotron was not asked for this request in this session'
        return None, disagreement_reason(read(agree)['agreement'])
    d = read(dst)['decision']
    if not d['attempt']:
        return None, f"not attempted: {d.get('local_reason')}; {'; '.join(d.get('disagreements', [])) or 'model and rule agree'}"
    return dst, None


def plan(session, stop=DEFAULT_STOP):
    if stop not in STOPS:
        raise SystemExit(f'--stop must be one of {STOPS}')
    rows, eps = [], []
    for e in read(session / 'declared_plan.json')['episodes']:
        row = dict(e, case=demo_plan.case_name(e['layout'], e['tag']))
        if e['order'] > stop:
            rows.append(row | {'planned': False, 'why': f'stopped at {stop} by plan'})
            continue
        path, why = spec_for(session, e)
        if path is None:
            rows.append(row | {'planned': False, 'why': why})
            continue
        sha = sha_file(path)
        eps.append({'tag': e['tag'], 'layout': e['layout'], 'spec': f'specs/{path.name}', 'spec_sha256': sha})
        rows.append(row | {'planned': True, 'spec': f'specs/{path.name}', 'spec_sha256': sha})
    if not eps:
        raise SystemExit('no declared episode up to the stop point can run; nothing to launch')
    vm_plan = {'schema': demo_plan.SCHEMA, 'session': session.name, 'stop_point': stop, 'label': LABEL, 'episodes': eps}
    path = session / f'vm_plan_stop{stop}.json'
    sha = its.write_json(path, vm_plan)
    demo_plan.check(path)
    its.write_json(session / f'plan_note_stop{stop}.json', {'plan': path.name, 'plan_sha256': sha, 'episodes': rows})
    return path, sha, rows


def vm_files(session):
    """{relative path: sha256} from vm/SHA256SUMS, after checking every listed file and that nothing else is there."""
    vm = session / 'vm'
    sums = {}
    for line in (vm / 'SHA256SUMS').read_text().splitlines():
        sha, rel = line.split(None, 1)
        sums[rel.strip().removeprefix('./').lstrip('*')] = sha
    bad = [r for r, want in sums.items() if not (vm / r).is_file() or sha_file(vm / r) != want]
    extra = sorted(str(p.relative_to(vm)) for p in vm.rglob('*') if p.is_file()
                   and str(p.relative_to(vm)) not in sums and p.name != 'SHA256SUMS')
    if bad or extra:
        raise SystemExit(f'copy-back check failed: mismatched or missing {bad}; not on the VM list {extra}')
    return sums


def received_instruction(h5):
    import h5py
    with h5py.File(h5, 'r') as f:
        got = f.attrs['instruction']
    return got.decode() if isinstance(got, bytes) else str(got)


def tsv(path):
    lines = Path(path).read_text().splitlines()
    head = lines[0].split('\t')
    return [dict(zip(head, x.split('\t'))) for x in lines[1:] if x.strip()]


def collect(session):
    sums = vm_files(session)
    vm = session / 'vm'
    sup = vm / SUPERVISOR
    launched = sha_file(sup / 'demo_plan.json')
    plans = {sha_file(p): p for p in session.glob('vm_plan_stop*.json')}
    if launched not in plans:
        raise SystemExit("the VM supervisor's demo_plan.json is none of this session's plans")
    note = read(session / f"plan_note_{plans[launched].stem.removeprefix('vm_plan_')}.json")
    exit_code = (sup / 'exit-code.txt').read_text().strip() if (sup / 'exit-code.txt').exists() else 'missing'
    ran = {r['case']: r for r in tsv(sup / 'episodes.tsv')}
    out = []
    for row in note['episodes']:
        res = {k: row[k] for k in ('order', 'tag', 'layout', 'request', 'source', 'case')}
        res['label'] = LABEL
        if not row['planned']:
            res.update(ran=False, why=row['why'])
        elif row['case'] not in ran:
            res.update(ran=False, why=f'planned, but the supervisor stopped before it (exit code {exit_code})')
        else:
            res.update(episode_in(session, row, ran[row['case']], sums))
        its.write_json(session / 'episodes' / f"{row['order']}_{row['tag']}" / 'result.json', res)
        out.append(res)
    summary = {'label': LABEL, 'plan': plans[launched].name, 'plan_sha256': launched, 'supervisor_exit_code': exit_code,
               'episodes': [{k: r.get(k) for k in ('order', 'tag', 'request', 'source', 'ran', 'why', 'holds', 'ended')}
                            for r in out]}
    its.write_json(session / 'episodes' / 'summary.json', summary)
    return out


def override_markers(log):
    """The wrapper's INTENT_OVERRIDE_APPLIED lines in one evaluation.log, parsed; recounted here on the Mac."""
    if not log.is_file():
        return []
    return [json.loads(x.split(' ', 1)[1]) for x in log.read_text(errors='replace').splitlines()
            if x.startswith('INTENT_OVERRIDE_APPLIED ')]


def grading_spec(session, row):
    """The spec whose success_check grades this episode: its own for Nemotron episodes; for a raw-words control, this
    session's agreed Nemotron spec for the same request (P1), never the hand-written prediction."""
    if row['source'] == 'literal':
        return session / 'specs' / f"{its.slug(row['request'])}.json"
    return session / row['spec']


def episode_in(session, row, tsv_row, sums):
    vm, case = session / 'vm', row['case']
    res = {'ran': True, 'supervisor_row': tsv_row}
    spec_copy = vm / 'runs' / case / 'spec.json'
    res['spec_sent_matches_plan'] = spec_copy.is_file() and sha_file(spec_copy) == row['spec_sha256']
    spec = read(session / row['spec'])
    want = spec['policy_instruction']
    out = vm / 'RoboLab/output' / case / TASK
    files = {r: s for r, s in sums.items() if r.startswith((f'runs/{case}/', f'RoboLab/output/{case}/'))}
    res['evidence'] = files
    h5 = out / 'synchronized.hdf5'
    res['policy_instruction_sent'] = want
    res['policy_received'] = received_instruction(h5) if h5.is_file() else None
    vm_check = vm / 'runs' / case / 'instruction-check.json'
    res['vm_instruction_check'] = read(vm_check) if vm_check.is_file() else None
    marks = override_markers(vm / 'runs' / case / 'evaluation.log')
    res['override_markers'] = marks
    # F1: the default sentence can equal Nemotron's, so only the marker proves the override ran
    res['instruction_checks'] = {
        'hdf5_instruction_equals_spec': res['policy_received'] == want,
        'one_override_marker_to_spec': len(marks) == 1 and marks[0].get('to') == want,
        'spec_sent_matches_plan': res['spec_sent_matches_plan'],
        'supervisor_failure_column_no': tsv_row.get('failure') == 'no',
        'vm_instruction_check_equals_spec': bool(res['vm_instruction_check'])
                                            and res['vm_instruction_check'].get('policy_received') == want}
    res['instruction_match'] = all(res['instruction_checks'].values())
    videos = sorted(p for p in out.glob('*.mp4') if not p.name.endswith('_viewport.mp4'))
    res['video'] = str(videos[0].relative_to(session)) if len(videos) == 1 else None
    trace = out / 'strict-metric.json'
    if not trace.is_file():
        return res | {'holds': None, 'ended': None, 'why': 'no trace: the episode did not finish its capture'}
    graded = grading_spec(session, row)
    rr = checker.load(trace)
    verdict = checker.evaluate(trace, read(graded)['success_check'], robot_rows=rr)
    res.update(checker=verdict, checker_sha256=sha_file(HERE / 'checker.py'), graded_with=str(graded.relative_to(session)),
               graded_with_sha256=sha_file(graded), ended=checker.describe_end(trace, robot_rows=rr),
               trace=str(trace.relative_to(session)))
    if not res['instruction_match']:
        failed = [k for k, ok in res['instruction_checks'].items() if not ok]
        return res | {'holds': None, 'graded': False,
                      'why': f"instruction not verified: received “{res['policy_received']}” / spec “{want}” ({', '.join(failed)})"}
    return res | {'holds': verdict.get('demo_holds'), 'graded': True}


# cards ------------------------------------------------------------------------------------------------------------------
LEFT_NOTE = "out of distribution; “left” is the robot's left (about image-left in this exterior view)"


def side_note(relation):
    if relation == 'left_of':
        return LEFT_NOTE
    if relation == 'right_of':
        return LEFT_NOTE.replace('left', 'right')
    if relation == 'beside':
        return 'out of distribution: the policy was trained on in-the-bowl tasks'
    return None


def spec_rows(model_spec, decision, robot_instruction, not_run_reason=None):
    m = model_spec
    feasible = m['feasible_for_this_robot']
    attempt = decision['attempt']
    if feasible:
        fe = ('feasible', 'true', 'ok')
    elif attempt:
        fe = ('feasible', 'false — attempted anyway', 'warn')
    else:
        fe = ('feasible', 'false — robot not run', 'bad')
    rows = [[('object', m['object'], None), ('relation', m['relation'], None), ('target', m['target'], None)],
            [('robot instruction', robot_instruction, None)], [fe]]
    note = side_note((m.get('success_check') or {}).get('relation'))
    if note and attempt:
        rows.append([('note', note, 'warn')])
    if not_run_reason:
        rows.append([('reason', not_run_reason, None)])
    return rows


def raw_json(m):
    return json.dumps({k: m[k] for k in ('object', 'target', 'relation', 'policy_instruction', 'feasible_for_this_robot',
                                         'attempt', 'success_check') if k in m}, indent=1)


def commanded_label(check):
    if check['reference'] == 'bowl' and check['relation'] in ('in', 'on'):
        return 'a commanded release (v5.1 rule)'
    return 'the gripper was commanded open during the episode'


def rest_label(v):
    d = v.get('rest_detail') or {}
    if d.get('pose_max_angular_rad_s') is None:
        return f"at rest, last 1 s (recorded {d.get('recorded_max_angular_rad_s', float('nan')):.2f} rad/s)"
    return (f"at rest, last 1 s · max {d['recorded_max_angular_rad_s']:.2f} rad/s recorded, "
            f"{d['pose_max_angular_rad_s']:.2f} from poses")


def checks(v, whose="Nemotron's check"):
    """The checker's itemised result for the verdict card: [label, true/false/null] items under [title, result, 'head']
    rows. Two graded heads (the spec's own check, then the demo floor) and one shown-only head."""
    if not v or 'conditions' not in v:
        return []
    sc, cond = v['success_check'], v['conditions']
    label = {'rest': rest_label(v), 'released': 'released for the last 1 s', 'commanded_release': commanded_label(sc)}
    out = [[whose, v['holds'], 'head'],
           [f"{sc['object']} {REL_PHRASE.get(sc['relation'], sc['relation'])} {sc['reference']} at the end", v['relation_at_end']]]
    out += [[label[k], cond.get(k)] for k in ('rest', 'released', 'commanded_release') if sc[f'require_{k}']]
    f = v['floor']
    out += [['Floor, every demo run', v['floor_holds'], 'head'], [label['commanded_release'], f['commanded_release']],
            [label['released'], f['released']], ['not already true at the start', f['not_true_at_start']]]
    shown = [] if sc['require_rest'] else [[label['rest'], cond['rest']]]
    if sc['relation'] in checker.GEOMETRIC:
        shown.append(['no gripper contact after the last eligible open', v.get('no_recontact_after_last_open')])
    if shown:
        out += [['Shown, not graded', None, 'head']] + shown
    return out


def verdict(v, ended, whose="Nemotron's check", why=None):
    """The end card's verdict for one checker result: green only when the spec's check and the floor both hold."""
    if v is None or v.get('demo_holds') is None:
        return {'ok': None, 'lines': ['not checkable'], 'detail': why or (v or {}).get('why', ''), 'checks': checks(v, whose)}
    check = v['success_check']
    rel, obj, ref = check['relation'], check['object'], check['reference']
    detail = (f"{whose[0].upper()}{whose[1:]}: {'holds' if v['holds'] else 'does not hold'}. Floor (commanded release, released, not true at "
              f"the start): {'holds' if v['floor_holds'] else 'does not hold'}. The recorded angular rate spikes; the "
              'rate from poses is shown beside it.')
    if v['demo_holds']:
        lines = [f'{obj} {REL_WORD.get(rel, rel)} {ref}']
    else:
        lines = [f'asked: {REL_WORD.get(rel, rel)} the {ref}', f'ended: {ended}']
    return {'ok': v['demo_holds'], 'lines': lines, 'detail': detail, 'checks': checks(v, whose)}


def unverified_verdict(r):
    items = [[k.replace('_', ' '), ok] for k, ok in r['instruction_checks'].items()]
    return {'ok': None, 'lines': ['instruction not verified'],
            'detail': f"received “{r.get('policy_received')}” / spec “{r['policy_instruction_sent']}”. "
                      "Not graded against Nemotron's check.",
            'checks': [['what GR00T was told', False, 'head']] + items}


def not_run_verdict(m, decision):
    return {'ok': None, 'lines': ['robot not run'], 'detail': f"Nemotron: {m['reason']}",
            'checks': [[f"Nemotron: attempt = {decision['nemotron_attempt']}", None],
                       [f"local rule: {decision['local_rule']}", None]]}


def disagreement_reason(agree):
    if not agree['all_valid']:
        return f"only {agree['valid']} of Nemotron's answers were valid specs; the robot needs all {REPEATS} valid and agreeing"
    differ = [k for k in its.AGREE_KEYS if not agree.get(k)]
    return f"Nemotron's {REPEATS} answers disagreed on {', '.join(differ)}; the robot runs only an answer given every time"


def video_dims(video):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                          '-of', 'csv=p=0', str(video)], capture_output=True, text=True, check=True)
    return tuple(map(int, out.stdout.strip().split(',')))


def exterior(video):
    """The exterior camera half of RoboLab's sensor video. Never the viewport video: no left/right claim rests on it."""
    if video_dims(video) != (1280, 360):
        raise SystemExit(f'{video} is not the 1280x360 RoboLab sensor video')
    return {'video': str(video), 'video_crop': EXTERIOR_CROP}


def footer(served, spec_sha, extra):
    return [f'Nemotron: {served} on Nebius Token Factory · spec sha256 {spec_sha[:12]}',
            f'{GROOT}', extra]


def plan_spec(session, r):
    """The spec file the episode was planned with (from the plan note)."""
    for p in session.glob('plan_note_stop*.json'):
        for row in read(p)['episodes']:
            if row['case'] == r['case'] and row.get('planned'):
                return session / row['spec']
    raise SystemExit(f"no planned spec for {r['case']}")


def served_models(spec):
    return ', '.join(sorted({c['served_model'] for c in spec['provenance']['calls'] if c.get('served_model')}))


def run_card(session, r):
    """One card per episode that started; with no footage (P3) the robot panel says so instead of playing."""
    spec_path = plan_spec(session, r)
    spec = read(spec_path)
    nemotron = r['source'] == 'nemotron'
    graded = session / r['graded_with'] if r.get('graded_with') else grading_spec(session, r)
    check = read(graded)['success_check'] if graded.exists() else spec.get('success_check')
    served = served_models(read(graded)) if graded.exists() else 'not called'
    if nemotron:
        rows = spec_rows(spec['nemotron'], spec['decision'], spec['policy_instruction'])
        raw = raw_json(spec['nemotron'])
        whose = "Nemotron's check"
    else:
        rows = [[('object', check['object'], None), ('relation', check['relation'], None), ('target', check['reference'], None)],
                [('robot instruction', spec['policy_instruction'], None)],
                [('source', "the user's own words, sent as-is; graded with this session's Nemotron check for the same "
                            'request', 'warn')]]
        raw = json.dumps({'policy_instruction (raw words)': spec['policy_instruction'],
                          'success_check (Nemotron, this session)': check}, indent=1)
        whose = "Nemotron's check (same request)"
        served = f'{served} (check only; the robot got the raw words)'
    if r.get('ran') and r.get('instruction_checks') and not r['instruction_match']:
        v = unverified_verdict(r)
    else:
        v = verdict(r.get('checker'), r.get('ended'), whose, r.get('why'))
    received = r.get('policy_received')
    label = (f"{r['case']} · GR00T received: “{received}”" if received and r.get('instruction_match')
             else f"{r['case']} · instruction not verified")
    if r.get('video'):
        robot = {**exterior(session / r['video']), 'robot_caption': "GR00T's exterior camera, as saved by the run"}
    else:
        robot = {'video': None, 'video_crop': None, 'robot_caption': 'no footage from this episode',
                 'no_footage_reason': r.get('why') or 'the episode started but saved no exterior video'}
    return {'kind': 'run', 'mockup': False, 'instruction': r['request'], 'nemotron_model': served,
            'spec_rows': rows, 'raw_json': raw, **robot, 'run_label': label,
            'verdict': v, 'footer': footer(served, sha_file(graded if graded.exists() else spec_path),
                                           f"checker {sha_file(HERE / 'checker.py')[:12]} · "
                                           f"episode {r['order']} of the declared order · MANIFEST.json"),
            'order': r['order']}


def not_run_card(session, slug, video, video_note):
    """A request the robot never attempted: Nemotron and the local rule said no, or the repeats disagreed (P2)."""
    nd = session / 'nemotron' / slug
    recs = [read(p) for p in sorted(nd.glob('r[0-9].json'))]
    if not recs:
        return None
    spec = session / 'specs' / f'{slug}.json'
    valid = [x for x in recs if x.get('nemotron')]
    if spec.exists():
        rec = read(spec)
        m = rec['nemotron']
        reason = m['reason'] if not m['feasible_for_this_robot'] else rec['decision'].get('local_reason', '')
        v, why = not_run_verdict(m, rec['decision']), 'Local rule: ' + rec['decision'].get('local_reason', '')
    else:
        agree = read(nd / 'agreement.json')['agreement']
        rec = valid[0] if valid else recs[0]
        m = rec.get('nemotron')
        reason = why = disagreement_reason(agree)
        v = {'ok': None, 'lines': ['robot not run'], 'detail': reason,
             'checks': [[f"valid answers: {agree['valid']}", agree['all_valid']]]
                       + [[f'same {k} every time', agree.get(k)] for k in its.AGREE_KEYS if agree['all_valid']]}
    served = ', '.join(sorted({c['served_model'] for x in recs for c in x['provenance']['calls'] if c.get('served_model')}))
    rows = (spec_rows(m, rec['decision'], m['policy_instruction'], reason) if m else
            [[('Nemotron', 'no valid answer', 'bad')], [('reason', reason, None)]])
    return {'kind': 'not_run', 'mockup': False, 'instruction': rec['input_text'], 'nemotron_model': served or 'no answer',
            'spec_rows': rows, 'raw_json': raw_json(m) if m else json.dumps({'errors': [x.get('error') for x in recs]}, indent=1),
            **exterior(video), 'robot_caption': video_note, 'not_run_reason': why,
            'run_label': 'no episode: the robot only runs specs that Nemotron gives every time and that both Nemotron '
                         'and the local rule call attemptable',
            'verdict': v,
            'footer': footer(served or 'no answer', its.sha_text(json.dumps(rec, indent=1) + '\n'), 'no checker run: nothing to grade')}


def write_cards(session):
    results = sorted((read(p) for p in session.glob('episodes/*/result.json')), key=lambda r: r['order'])
    if not results:
        raise SystemExit('no episodes/*/result.json: run collect first')
    written = []
    for r in results:
        if r['ran']:   # P3: every episode that started gets a card, footage or not
            written.append((f"{r['order']}_{r['tag']}", run_card(session, r)))
    first = next((session / r['video'] for r in results if r.get('video')), None)
    note = ('layout 4004 at the start, frame 0 of this session\'s first episode' if first else
            'layout 4004 at the start, from step 8 before-phase footage')
    video = first or next(BEFORE_4004.glob('Pick_up*_0.mp4'))
    for req in read(session / 'declared_plan.json')['requests']:
        slug = its.slug(req['text'])
        if (session / 'specs' / f'{slug}.json').exists() and read(session / 'specs' / f'{slug}.json')['decision']['attempt']:
            continue
        card = not_run_card(session, slug, video, note)
        if card:
            written.append((f'notrun_{slug}', card))
    for name, card in written:
        its.write_json(session / 'cards' / f'{name}.json', card)
    its.write_json(session / 'cards' / 'summary.json', summary_card(session, results))
    return [n for n, _ in written]


def summary_card(session, results, mockup=False):
    rows = []
    for r in results:
        if r['ran'] and r.get('instruction_checks') and not r['instruction_match']:
            mark, text = 'none', f"instruction not verified: received “{r.get('policy_received')}”"
        elif r['ran'] and r.get('holds') is not None:
            sc = r['checker']['success_check']
            mark = 'ok' if r['holds'] else 'bad'
            text = (f"{sc['object']} {REL_WORD.get(sc['relation'], sc['relation'])} {sc['reference']}" if r['holds']
                    else f"asked {REL_WORD.get(sc['relation'], sc['relation'])} the {sc['reference']}, ended {r.get('ended')}")
        elif r['ran']:
            mark, text = 'none', r.get('why') or 'not checkable'
        else:
            mark, text = 'none', 'not run, ' + r['why']
        src = 'via Nemotron' if r['source'] == 'nemotron' else "raw words, graded with Nemotron's check"
        rows.append({'order': r['order'], 'request': r['request'], 'source': src, 'layout': r['layout'], 'mark': mark,
                     'text': text})
    for req in read(session / 'declared_plan.json')['requests']:
        slug = its.slug(req['text'])
        spec = session / 'specs' / f'{slug}.json'
        if any(r['request'] == req['text'] for r in results if r['ran']):
            continue
        if spec.exists() and not read(spec)['decision']['attempt']:
            text = 'robot not run: Nemotron and the local rule say do not attempt'
        elif not spec.exists() and (session / 'nemotron' / slug / 'agreement.json').exists():
            text = 'robot not run: ' + disagreement_reason(read(session / 'nemotron' / slug / 'agreement.json')['agreement'])
        else:
            continue
        rows.append({'order': '\u2013', 'request': req['text'], 'source': 'via Nemotron', 'layout': 4004, 'mark': 'none',
                     'text': text})
    return {'kind': 'summary', 'mockup': mockup, 'title': 'Every episode, in the declared order',
            'subtitle': f'declared {read(session / "declared_plan.json")["declared_utc"]} · session {session.name} · '
                        'nothing rerun, nothing dropped · each request run twice under the same setup', 'rows': rows}


# manifest ---------------------------------------------------------------------------------------------------------------
def manifest(session):
    files = {str(p.relative_to(session)): sha_file(p) for p in sorted(session.rglob('*'))
             if p.is_file() and p.name != 'MANIFEST.json'}
    snapshot_drift = [n for n in CODE if (session / 'snapshot' / n).exists()
                      and sha_file(session / 'snapshot' / n) != sha_file(HERE / n)]
    requests = []
    for req in read(session / 'declared_plan.json')['requests']:
        slug = its.slug(req['text'])
        nd = session / 'nemotron' / slug
        recs = [read(p) for p in sorted(nd.glob('r[0-9].json'))]
        spec = session / 'specs' / f'{slug}.json'
        requests.append({
            'text': req['text'], 'input_file': f'nemotron/{slug}/input.txt' if (nd / 'input.txt').exists() else None,
            'calls': [c for r in recs for c in r['provenance']['calls']],
            'served_models': sorted({c.get('served_model') for r in recs for c in r['provenance']['calls'] if c.get('served_model')}),
            'usage_total': {k: sum((c.get('usage') or {}).get(k, 0) for r in recs for c in r['provenance']['calls'])
                            for k in ('prompt_tokens', 'completion_tokens')},
            'agreement': read(nd / 'agreement.json')['agreement'] if (nd / 'agreement.json').exists() else None,
            'robot_spec': {'file': f'specs/{slug}.json', 'sha256': files.get(f'specs/{slug}.json')} if spec.exists() else None,
            'decision': read(spec)['decision'] if spec.exists() else (recs[0].get('decision') if recs else None)})
    episodes = []
    for p in sorted(session.glob('episodes/*/result.json'), key=lambda p: read(p)['order']):
        r = read(p)
        name = f"{r['order']}_{r['tag']}"
        clip = f'clips/{name}.mp4'
        episodes.append({'order': r['order'], 'tag': r['tag'], 'case': r['case'], 'request': r['request'], 'source': r['source'],
                         'ran': r['ran'], 'why_not_run': r.get('why') if not r['ran'] else None,
                         'result': {'file': str(p.relative_to(session)), 'sha256': files[str(p.relative_to(session))]},
                         'policy_instruction_sent': r.get('policy_instruction_sent'), 'policy_received': r.get('policy_received'),
                         'instruction_match': r.get('instruction_match'), 'holds': r.get('holds'), 'ended': r.get('ended'),
                         'evidence_files_checked': len(r.get('evidence', {})),
                         'clip': {'file': clip, 'sha256': files.get(clip)} if clip in files else None})
    m = {'schema': 'intent-demo-manifest-v1', 'label': LABEL, 'session': session.name,
         'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'declared_plan': {'file': 'declared_plan.json', 'sha256': files['declared_plan.json'], 'pinned': DECLARED_SHA256},
         'system_prompt': {'file': f'snapshot/{its.PROMPT_FILE.name}', 'sha256': files.get(f'snapshot/{its.PROMPT_FILE.name}'),
                           'pinned': its.PROMPT_SHA256},
         'code_changed_since_session_opened': snapshot_drift,
         'robot': GROOT, 'requests': requests, 'episodes': episodes,
         'reel': {'file': 'clips/reel.mp4', 'sha256': files.get('clips/reel.mp4')}, 'files': files}
    path = session / 'MANIFEST.json'
    path.write_text(json.dumps(m, indent=1) + '\n')   # the only file rewritten in place: it is regenerated, never edited
    return m


# mockup -----------------------------------------------------------------------------------------------------------------
MOCKUP_NOTE = ('MOCKUP — hand-written spec, Nemotron not called; footage: step 8 before-phase run A/P0/4004, '
               'where GR00T was told “Pick up the banana and place it in the bowl”')


def mockup(out):
    """Cards for the three requests from the hand-written expected specs, on the before-phase A/P0/4004 run (read-only)."""
    out = Path(out).resolve()
    if out.is_relative_to(EVIDENCE.resolve()) or out.is_relative_to(HERE.parents[1] / 'experiments/bowl_step8'):
        raise SystemExit('refusing to write into sealed evidence')
    out.mkdir(parents=True, exist_ok=True)
    expected = {k: v for k, v in read(HERE / 'fixtures/expected_specs.json').items() if not k.startswith('_')}
    trace = BEFORE_4004 / 'strict-metric.json'
    video = next(BEFORE_4004.glob('Pick_up*_0.mp4'))
    rr = checker.load(trace)
    ended = checker.describe_end(trace, robot_rows=rr)
    served = 'nvidia/Nemotron-3_5-Lightning (MOCKUP: not called)'
    made = {}
    for text, m in expected.items():
        decision = its.decide(m)
        base = {'mockup': True, 'mockup_note': MOCKUP_NOTE, 'instruction': text, 'nemotron_model': served,
                'raw_json': raw_json(m), **exterior(video)}
        sha = its.sha_text(json.dumps(m, sort_keys=True))
        if decision['attempt']:
            v = checker.evaluate(trace, m['success_check'], robot_rows=rr)
            card = base | {'kind': 'run', 'spec_rows': spec_rows(m, decision, m['policy_instruction']),
                           'robot_caption': "GR00T's exterior camera, as saved by the run",
                           'run_label': 'step8_A_P0_l4004_s0_v1 · GR00T received: “Pick up the banana and place it in the bowl”',
                           'verdict': verdict(v, ended, "the hand-written spec's check"), 'checker': v}
            card['footer'] = footer(served, sha, f"checker {sha_file(HERE / 'checker.py')[:12]} on the real before-phase trace · MOCKUP")
        else:
            card = base | {'kind': 'not_run', 'spec_rows': spec_rows(m, decision, m['policy_instruction'], m['reason']),
                           'robot_caption': 'layout 4004 at the start, from step 8 before-phase footage',
                           'not_run_reason': 'Local rule: ' + decision['local_reason'],
                           'run_label': 'no episode: the robot only runs specs that Nemotron and the local rule both call attemptable',
                           'verdict': not_run_verdict(m, decision),
                           'footer': footer(served, sha, 'no checker run: nothing to grade · MOCKUP')}
        name = 'MOCKUP_' + its.slug(text)
        (out / f'{name}.json').write_text(json.dumps(card, indent=1) + '\n')
        made[name] = card
    rows = [{'order': e['order'], 'request': e['request'], 'layout': e['layout'], 'mark': 'none',
             'source': 'via Nemotron' if e['source'] == 'nemotron' else "raw words, graded with Nemotron's check",
             'text': 'verdict appears here after the run' if e['order'] <= DEFAULT_STOP
             else f'not run, stopped at {DEFAULT_STOP} by plan'} for e in read(DECLARED)['episodes']]
    summary = {'kind': 'summary', 'mockup': True, 'mockup_note': 'MOCKUP \u2014 layout of the closing card; no demo episode has run',
               'title': 'Every episode, in the declared order',
               'subtitle': f"declared {read(DECLARED)['declared_utc']} \u00b7 nothing rerun, nothing dropped \u00b7 each request "
                           'run twice under the same setup', 'rows': rows}
    (out / 'MOCKUP_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    made['MOCKUP_summary'] = summary
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('new')
    for name in ('compile', 'collect', 'cards', 'manifest'):
        sub.add_parser(name).add_argument('session')
    p = sub.add_parser('plan')
    p.add_argument('session')
    p.add_argument('--stop', type=int, default=DEFAULT_STOP, choices=STOPS)
    sub.add_parser('mockup').add_argument('out')
    a = ap.parse_args(argv)
    if a.cmd == 'new':
        print(new())
    elif a.cmd == 'mockup':
        print('\n'.join(mockup(a.out)))
    else:
        s = session_dir(a.session)
        if a.cmd == 'compile':
            for r in compile_requests(s):
                print(json.dumps({k: r.get(k) for k in ('request', 'spec_file', 'policy_instruction', 'decision')}))
            print(f'demo spent ${its.demo_spent():.6f} of ${its.DEMO_CAP_USD:.2f}')
        elif a.cmd == 'plan':
            path, sha, rows = plan(s, a.stop)
            for r in rows:
                print(f"{r['order']}\t{r['case']}\t{'RUN' if r['planned'] else 'NOT RUN: ' + r['why']}")
            print(f'PLAN {path.relative_to(its.RUNS.parent)} {sha}')
        elif a.cmd == 'collect':
            for r in collect(s):
                print(json.dumps({k: r.get(k) for k in ('order', 'case', 'ran', 'why', 'instruction_match', 'holds', 'ended')}))
        elif a.cmd == 'cards':
            print('\n'.join(write_cards(s)))
        else:
            m = manifest(s)
            print(f"MANIFEST {len(m['files'])} files; code changed since the session opened: {m['code_changed_since_session_opened']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
