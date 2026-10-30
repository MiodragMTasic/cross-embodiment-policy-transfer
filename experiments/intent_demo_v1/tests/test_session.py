"""The proof bundle: session, compile (mocked Token Factory), plan, copy-back check, collect, cards, manifest, mockup."""
import copy
import hashlib
import json
import shutil

import h5py
import pytest

import demo_plan
import demo_session as ds
import intent_to_spec as its
import render_clip
from conftest import EVIDENCE, EXPECTED, TASK, a_trace, mock_client, needs_evidence, response

FRUIT, LEFT, BOWL = 'put the fruit away', 'put the banana beside the bowl, on its left', 'pick up the bowl and put it on the banana'


def replies(left=None):
    """Three canned replies per declared request, in the declared order."""
    lefts = left or [EXPECTED[LEFT]] * 3
    return [response(json.dumps(s)) for s in [EXPECTED[FRUIT]] * 3 + lefts + [EXPECTED[BOWL]] * 3]


@pytest.fixture
def session(sandbox):
    s = ds.new()
    client, calls = mock_client(replies())
    ds.compile_requests(s, client)
    assert len(calls) == 9
    return s


def test_declared_plan_is_pinned_and_consistent():
    d = ds.read(ds.DECLARED)
    assert hashlib.sha256(ds.DECLARED.read_bytes()).hexdigest() == ds.DECLARED_SHA256
    assert [e['order'] for e in d['episodes']] == [1, 2, 3, 4, 5, 6] and {e['layout'] for e in d['episodes']} == {4004}
    texts = [r['text'] for r in d['requests']]
    assert texts == [FRUIT, LEFT, BOWL] and all(e['request'] in texts for e in d['episodes'])
    assert [e['request'] for e in d['episodes'][:2]] == [FRUIT, LEFT]   # the cheapest useful stop has one of each
    for name, sha in ds.LITERAL_SHA256.items():
        assert hashlib.sha256((ds.HERE / 'specs' / name).read_bytes()).hexdigest() == sha


def test_new_refuses_a_changed_declaration(sandbox, monkeypatch):
    monkeypatch.setattr(ds, 'DECLARED_SHA256', '0' * 64)
    with pytest.raises(SystemExit, match='changed after it was declared'):
        ds.new()


def test_compile_saves_every_body_and_the_robot_spec(session):
    nd = session / 'nemotron' / its.slug(LEFT)
    assert (nd / 'input.txt').read_text() == LEFT
    rec = ds.read(nd / 'r1.json')
    for c in rec['provenance']['calls']:
        body = (nd / c['request_file']).read_text()
        assert hashlib.sha256(body.encode()).hexdigest() == c['request_file_sha256']
        assert json.loads((nd / c['response_file']).read_text())['model'] == c['served_model']
    assert 'sk-FAKE' not in ''.join(p.read_text() for p in session.rglob('*.json'))
    spec = ds.read(session / 'specs' / f'{its.slug(LEFT)}.json')
    assert spec['chosen'] == 'r1 of 3' and spec['agreement']['all_agree'] and spec['decision']['attempt']
    bowl = ds.read(session / 'specs' / f'{its.slug(BOWL)}.json')
    assert not bowl['decision']['attempt']   # agreed, but not attempted: no robot
    assert (session / 'snapshot' / its.PROMPT_FILE.name).read_text() == its.PROMPT_FILE.read_text()


def test_plan_takes_the_declared_order_up_to_the_stop(session):
    path, sha, rows = ds.plan(session, 2)
    p = ds.read(path)
    assert [e['tag'] for e in p['episodes']] == ['fruitaway1', 'leftbeside1']
    assert [r['planned'] for r in rows] == [True, True, False, False, False, False]
    assert rows[2]['why'] == 'stopped at 2 by plan'
    assert [r[0] for r in demo_plan.check(path)] == ['intentdemo_A_P0_l4004_fruitaway1_v1', 'intentdemo_A_P0_l4004_leftbeside1_v1']
    path6, _, rows6 = ds.plan(session, 6)
    assert len(ds.read(path6)['episodes']) == 6 and all(r['planned'] for r in rows6)
    assert (session / 'specs' / 'literal_put_the_fruit_away.json').is_file()
    with pytest.raises(PermissionError):
        ds.plan(session, 2)   # each plan is written once


@needs_evidence
def test_default_stop_is_4_and_the_card_says_so(session, capsys):
    assert ds.main(['plan', str(session)]) == 0
    assert 'vm_plan_stop4.json' in capsys.readouterr().out
    path = session / 'vm_plan_stop4.json'
    assert [e['tag'] for e in ds.read(path)['episodes']] == ['fruitaway1', 'leftbeside1', 'fruitaway2', 'leftbeside2']
    fake_copy_back(session, path)
    res = ds.collect(session)
    rows = ds.summary_card(session, res)['rows']
    assert [r['text'] for r in rows[4:6]] == ['not run, stopped at 4 by plan'] * 2
    assert [r['source'] for r in rows[4:6]] == ["raw words, graded with Nemotron's check"] * 2


def test_disagreeing_repeats_keep_the_robot_off(sandbox):
    s = ds.new()
    other = copy.deepcopy(EXPECTED[LEFT])
    other['policy_instruction'] = 'Place the banana to the left of the bowl'
    client, _ = mock_client(replies(left=[EXPECTED[LEFT], other, EXPECTED[LEFT]]))
    ds.compile_requests(s, client)
    assert not (s / 'specs' / f'{its.slug(LEFT)}.json').exists()
    _, _, rows = ds.plan(s, 4)
    assert [r['planned'] for r in rows[:4]] == [True, False, True, False]
    assert rows[1]['why'].startswith("Nemotron's 3 answers disagreed on policy_instruction")
    _, _, rows6 = ds.plan(s, 6)   # P1: the raw-words left control has no Nemotron check to be graded with
    assert rows6[4]['planned'] is True and rows6[5]['planned'] is False and 'raw-words control' in rows6[5]['why']


def fake_copy_back(session, plan_path, received=None, ran=None, markers=None, no_video=()):
    """A VM copy-back built from the before-phase A/P0/4004 run, standing in for the demo episodes. The wrapper's
    INTENT_OVERRIDE_APPLIED marker is written when the override took (received is None), unless markers says otherwise."""
    vm = session / 'vm'
    sup = vm / ds.SUPERVISOR
    sup.mkdir(parents=True)
    shutil.copyfile(plan_path, sup / 'demo_plan.json')
    rows = ['phase\trobot\tpolicy\tlayout\tcase\tstatus\tfailure\tv4_success\tinstruction_ok\tseconds']
    src = EVIDENCE / 'RoboLab/output/step8_A_P0_l4004_s0_v1' / TASK
    for case, layout, spec_path, sha in demo_plan.check(plan_path)[:ran]:
        spec = ds.read(spec_path)
        run = vm / 'runs' / case
        run.mkdir(parents=True)
        shutil.copyfile(spec_path, run / 'spec.json')
        got = received or spec['policy_instruction']
        (run / 'instruction-check.json').write_text(json.dumps({'policy_received': got}))
        n = (0 if received else 1) if markers is None else markers
        mark = json.dumps({'from': 'Pick up the banana and place it in the bowl', 'to': spec['policy_instruction']})
        (run / 'evaluation.log').write_text('starting\n' + f'INTENT_OVERRIDE_APPLIED {mark}\n' * n + 'done\n')
        out = vm / 'RoboLab/output' / case / TASK
        out.mkdir(parents=True)
        shutil.copyfile(src / 'strict-metric.json', out / 'strict-metric.json')
        if case not in no_video:
            shutil.copyfile(next(src.glob('Pick_up*_0.mp4')), out / (got.replace(' ', '_') + '_0.mp4'))
        with h5py.File(out / 'synchronized.hdf5', 'w') as f:
            f.attrs['instruction'] = got
        rows.append(f'intentdemo_v1\tA\tP0\t{layout}\t{case}\t0\tno\tTrue\tyes\t70')
    (sup / 'episodes.tsv').write_text('\n'.join(rows) + '\n')
    (sup / 'exit-code.txt').write_text('0\n')
    sums = [f'{ds.sha_file(p)}  {p.relative_to(vm)}' for p in sorted(vm.rglob('*')) if p.is_file()]
    (vm / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    return vm


@needs_evidence
def test_collect_cards_manifest_end_to_end(session):
    path, _, _ = ds.plan(session, 2)
    fake_copy_back(session, path)
    res = ds.collect(session)
    fruit, left = res[0], res[1]
    assert fruit['ran'] and fruit['instruction_match'] and fruit['spec_sent_matches_plan']
    assert fruit['holds'] is True and fruit['ended'] == 'in the bowl'
    assert left['holds'] is False and left['ended'] == 'in the bowl' and left['checker']['relation_at_end'] is False
    assert [r['ran'] for r in res] == [True, True, False, False, False, False]
    names = ds.write_cards(session)
    assert names == ['1_fruitaway1', '2_leftbeside1', f'notrun_{its.slug(BOWL)}']
    card = ds.read(session / 'cards' / '2_leftbeside1.json')
    assert card['verdict']['ok'] is False and card['verdict']['lines'] == ['asked: LEFT OF the bowl', 'ended: in the bowl']
    assert card['video_crop'] == [640, 360, 0, 0] and not card['mockup']
    assert 'GR00T received' in card['run_label'] and EXPECTED[LEFT]['policy_instruction'] in card['run_label']
    nr = ds.read(session / 'cards' / f'notrun_{its.slug(BOWL)}.json')
    assert nr['kind'] == 'not_run' and nr['verdict']['lines'] == ['robot not run']
    summary = ds.read(session / 'cards' / 'summary.json')
    assert [r['mark'] for r in summary['rows']] == ['ok', 'bad', 'none', 'none', 'none', 'none', 'none']
    assert summary['rows'][0]['text'] == 'banana IN bowl' and summary['rows'][-1]['request'] == BOWL
    m = ds.manifest(session)
    fc = ds.read(session / 'cards' / '1_fruitaway1.json')['verdict']
    heads = [c[0] for c in fc['checks'] if len(c) > 2]
    assert heads == ["Nemotron's check", 'Floor, every demo run'] and fc['ok'] is True   # F2: both shown
    assert [c[1] for c in card['verdict']['checks'] if len(c) > 2][:2] == [False, True]   # left: check fails, floor holds
    assert ['the gripper was commanded open during the episode', True] in card['verdict']['checks']   # P4 label
    assert ['no gripper contact after the last eligible open', True] in card['verdict']['checks']
    assert any('note' in c[0] and 'robot' in c[1] for row in card['spec_rows'] for c in row)   # robot's left, OOD
    assert m['declared_plan']['sha256'] == ds.DECLARED_SHA256 and m['system_prompt']['sha256'] == its.PROMPT_SHA256
    assert m['code_changed_since_session_opened'] == []
    assert [e['holds'] for e in m['episodes'][:2]] == [True, False] and m['episodes'][0]['evidence_files_checked'] == 6
    assert all(ds.sha_file(session / f) == h for f, h in m['files'].items())
    req = {r['text']: r for r in m['requests']}
    assert req[FRUIT]['usage_total'] == {'prompt_tokens': 4500, 'completion_tokens': 600}
    assert req[BOWL]['robot_spec'] and not req[BOWL]['decision']['attempt']


@needs_evidence
def test_collect_rejects_tampered_or_extra_files(session):
    path, _, _ = ds.plan(session, 2)
    vm = fake_copy_back(session, path)
    (vm / ds.SUPERVISOR / 'exit-code.txt').write_text('1\n')
    with pytest.raises(SystemExit, match='copy-back check failed'):
        ds.collect(session)
    (vm / ds.SUPERVISOR / 'exit-code.txt').write_text('0\n')
    (vm / 'stray.txt').write_text('x')
    with pytest.raises(SystemExit, match='not on the VM list'):
        ds.collect(session)


@needs_evidence
def test_wrong_instruction_is_reported_not_hidden(session):
    """F1: the fruit spec's sentence equals the default, so the hdf5 alone matches; with no override marker the run
    is still not verified, gets no mark, and its card says what was received."""
    path, _, _ = ds.plan(session, 2)
    fake_copy_back(session, path, received='Pick up the banana and place it in the bowl')
    res = ds.collect(session)
    fruit, left = res[0], res[1]
    assert fruit['instruction_checks']['hdf5_instruction_equals_spec'] is True
    assert fruit['instruction_checks']['one_override_marker_to_spec'] is False
    assert fruit['instruction_match'] is False and fruit['holds'] is None and not fruit['graded']
    assert fruit['checker']['demo_holds'] is True   # the physics passed, but it is not graded as Nemotron's run
    assert left['instruction_match'] is False and left['policy_received'] == 'Pick up the banana and place it in the bowl'
    ds.write_cards(session)
    for name, r in (('1_fruitaway1', fruit), ('2_leftbeside1', left)):
        card = ds.read(session / 'cards' / f'{name}.json')
        assert card['verdict']['ok'] is None and card['verdict']['lines'] == ['instruction not verified']
        assert f"received “{r['policy_received']}” / spec “{r['policy_instruction_sent']}”" in card['verdict']['detail']
        assert card['run_label'].endswith('instruction not verified')
    rows = ds.read(session / 'cards' / 'summary.json')['rows']
    assert [r['mark'] for r in rows[:2]] == ['none', 'none'] and rows[0]['text'].startswith('instruction not verified')


@needs_evidence
def test_two_markers_or_a_failed_supervisor_row_are_not_verified(session):
    path, _, _ = ds.plan(session, 2)
    fake_copy_back(session, path, markers=2)
    res = ds.collect(session)
    assert [r['instruction_match'] for r in res[:2]] == [False, False]
    assert res[0]['instruction_checks']['one_override_marker_to_spec'] is False and res[0]['holds'] is None


@needs_evidence
def test_literal_control_is_graded_with_this_sessions_nemotron_check(session):
    """P1: the raw-words episode is graded with this session's agreed Nemotron spec, not the hand-written file."""
    path, _, _ = ds.plan(session, 6)
    fake_copy_back(session, path)
    res = ds.collect(session)
    raw = res[4]
    assert raw['source'] == 'literal' and raw['graded_with'] == f'specs/{its.slug(FRUIT)}.json'
    nem = ds.read(session / 'specs' / f'{its.slug(FRUIT)}.json')['success_check']
    lit = ds.read(session / 'specs' / f'literal_{its.slug(FRUIT)}.json')['success_check']
    assert raw['checker']['success_check'] == nem and nem != lit   # the fixture's rim_counts differs: proves which one
    ds.write_cards(session)
    card = ds.read(session / 'cards' / '5_rawfruit1.json')
    assert "graded with this session's Nemotron check" in card['spec_rows'][2][0][1]
    assert json.loads(card['raw_json'])['success_check (Nemotron, this session)'] == nem


@needs_evidence
def test_episode_without_footage_still_gets_a_card(session):
    """P3: started, no video saved: a card with a no-footage panel, never a silent gap."""
    path, _, _ = ds.plan(session, 2)
    fake_copy_back(session, path, no_video=('intentdemo_A_P0_l4004_leftbeside1_v1',))
    res = ds.collect(session)
    assert res[1]['ran'] and res[1]['video'] is None
    names = ds.write_cards(session)
    assert '2_leftbeside1' in names
    card = ds.read(session / 'cards' / '2_leftbeside1.json')
    assert card['video'] is None and card['robot_caption'] == 'no footage from this episode'
    clip = render_clip.make(card)
    img = clip.frame(clip.duration - .1)
    clip.close()
    assert img.size == (1920, 1080)


def test_disagreement_card_gives_the_real_reason(sandbox):
    """P2: the card and summary read agreement.json."""
    s = ds.new()
    other = copy.deepcopy(EXPECTED[LEFT])
    other['policy_instruction'] = 'Place the banana to the left of the bowl'
    client, _ = mock_client(replies(left=[EXPECTED[LEFT], other, EXPECTED[LEFT]]))
    ds.compile_requests(s, client)
    card = ds.not_run_card(s, its.slug(LEFT), next(ds.BEFORE_4004.glob('Pick_up*_0.mp4')), 'start')
    assert card['verdict']['detail'] == "Nemotron's 3 answers disagreed on policy_instruction; the robot runs only an answer given every time"
    assert ['same policy_instruction every time', False] in card['verdict']['checks']
    rows = ds.summary_card(s, [])['rows']
    assert any(r['request'] == LEFT and 'disagreed on policy_instruction' in r['text'] for r in rows)


def test_nemotron_is_asked_once_per_demo(sandbox):
    """F3: a second compile, or a second session after one asked, is refused before any call."""
    s = ds.new()
    client, calls = mock_client(replies())
    ds.compile_requests(s, client)
    assert len(calls) == 9
    with pytest.raises(SystemExit, match='already asked Nemotron'):
        ds.new()
    client2, calls2 = mock_client(replies())
    with pytest.raises(SystemExit, match=f'{its.PURPOSE} row'):
        ds.compile_requests(s, client2)
    assert calls2 == []


def test_compile_refuses_when_any_ledger_has_a_demo_row(sandbox):
    s = ds.new()
    its.SEALED_LEDGER.write_text(json.dumps({'purpose': f'{its.PURPOSE}:old:r1', 'usd': 0.0001}) + '\n')
    client, calls = mock_client(replies())
    with pytest.raises(SystemExit, match=f'{its.PURPOSE} row'):
        ds.compile_requests(s, client)
    assert calls == []


@needs_evidence
def test_halted_supervisor_lists_the_rest_as_not_run(session):
    path, _, _ = ds.plan(session, 4)
    fake_copy_back(session, path, ran=1)
    res = ds.collect(session)
    assert [r['ran'] for r in res[:4]] == [True, False, False, False]
    assert 'stopped before it' in res[1]['why'] and res[4]['why'] == 'stopped at 4 by plan'


@needs_evidence
def test_mockup_cards_render(tmp_path):
    made = ds.mockup(tmp_path)
    assert set(made) == {f'MOCKUP_{its.slug(t)}' for t in (FRUIT, LEFT, BOWL)} | {'MOCKUP_summary'}
    assert all(c['mockup'] for c in made.values())
    fruit = made[f'MOCKUP_{its.slug(FRUIT)}']
    assert fruit['verdict']['ok'] is True and 'not called' in fruit['nemotron_model'] and 'MOCKUP' in fruit['mockup_note']
    assert made[f'MOCKUP_{its.slug(LEFT)}']['verdict']['ok'] is False
    for name in (f'MOCKUP_{its.slug(FRUIT)}', 'MOCKUP_summary'):
        clip = render_clip.make(made[name])
        img = clip.frame(clip.duration - .1)
        clip.close()
        assert img.size == (1920, 1080)


def test_mockup_refuses_the_evidence_folder():
    with pytest.raises(SystemExit, match='sealed'):
        ds.mockup(EVIDENCE / 'x')


@needs_evidence
def test_make_clips_end_to_end(session, monkeypatch):
    """The shell path after copy-back: cards, clips, summary, reel, MANIFEST. Renders about 45 s of 1080p video."""
    import subprocess
    path, _, _ = ds.plan(session, 2)
    fake_copy_back(session, path)
    ds.collect(session)
    env = dict(__import__('os').environ, INTENT_DEMO_RUNS=str(its.RUNS))
    out = subprocess.run(['bash', str(ds.HERE / 'make_clips.sh'), str(session)], env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    clips = session / 'clips'
    names = ['1_fruitaway1', '2_leftbeside1', f'notrun_{its.slug(BOWL)}', 'summary']
    for n in names:
        assert (clips / f'{n}.mp4').stat().st_size > 50_000 and (clips / f'{n}.png').is_file()
    lengths = {n: float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0',
                                                 str(clips / f'{n}.mp4')], text=True)) for n in names + ['reel']}
    assert all(10 <= lengths[n] <= 20 for n in names[:3]) and lengths['summary'] == pytest.approx(6, abs=.1)
    assert lengths['reel'] == pytest.approx(sum(lengths[n] for n in names), abs=.5)
    m = ds.read(session / 'MANIFEST.json')
    assert m['reel']['sha256'] == ds.sha_file(clips / 'reel.mp4')
    assert [e['clip']['file'] for e in m['episodes'][:2]] == ['clips/1_fruitaway1.mp4', 'clips/2_leftbeside1.mp4']
    again = subprocess.run(['bash', str(ds.HERE / 'make_clips.sh'), str(session)], env=env, capture_output=True, text=True)
    assert again.returncode == 3 and 'made once' in again.stdout


def test_earlier_prompt_version_session_does_not_block_new(sandbox):
    """Round 3 B1: a session that asked with another prompt version blocks neither new() nor compile's session check."""
    old = its.RUNS / '20261025T013018Z'
    (old / 'nemotron' / 'put_the_fruit_away').mkdir(parents=True)
    (old / 'snapshot').mkdir()
    (old / 'snapshot' / 'intent_to_spec_v1.txt').write_text('v1')
    assert ds.asked_sessions() == []
    s = ds.new()
    assert (s / 'snapshot' / its.PROMPT_FILE.name).is_file()
    (s / 'nemotron').mkdir()
    assert ds.asked_sessions() == [s.name]
