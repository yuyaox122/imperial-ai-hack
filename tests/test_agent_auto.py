from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

from agent import data, jobs, llm, loop, research
from agent.auto import run_auto
from agent.claims import Claim
from sense import models
from sense.contracts import FitRequest, ModelInfo, ReportCard, TraceEvent

CAND = data.DatasetCandidate('o/d', 'Sounds', 'https://huggingface.co/datasets/o/d', 'cc-by', 120)

@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv('SENSE_FAKE', '1')
    monkeypatch.setenv('SENSE_HOME', str(tmp_path / 'home'))
    w = SimpleNamespace(events=[], candidates=[], prewired='gtzan', research_error=None,
                        fetch_error=None, build_error=None, fetch_calls=[], build_calls=[], fit_calls=[],
                        claim_status={}, origin='fake')
    def search(*args, **kwargs):
        if w.research_error:
            raise w.research_error
        return research.Research({'specialists': [
            {'hub_id': 'laion/clap-htsat-unfused', 'statement': 'recognises sounds', 'source_url': 'https://example.org/clap'},
            {'hub_id': 'other/model', 'statement': 'recognises sounds', 'source_url': 'https://example.org/other'}]}, [], w.origin)
    def find(*args):
        return w.candidates, [Claim(c.hub_id, 'dataset', 'labelled sounds', c.source_url,
                                   status=w.claim_status.get(c.hub_id, 'unverified')) for c in w.candidates], 'live'
    def fetch(candidate, dest, scope):
        w.fetch_calls.append((candidate, dest, scope))
        if w.fetch_error:
            raise w.fetch_error
        return data.FetchReport(dest, candidate.hub_id, dict.fromkeys(['a', 'b', 'c', 'd'], 30), 120, 1000, candidate.licence, 500)
    def build(folder, name, **kwargs):
        w.build_calls.append(dict(folder=folder, name=name, **kwargs))
        if w.build_error:
            raise w.build_error
        return name
    def fit(request, emit, planner=None, *, on_cards=None, t0=None):
        w.fit_calls.append(request)
        card = ReportCard(request.model, request.modality, request.dataset, request.scope,
                          'vector:x', 'vector', 120, .25, .9, {'specialist_alone': .9})
        emit(TraceEvent('verify', 'ok', 'verified', {}, round(time.monotonic() - t0, 3)))
        on_cards([card])
        emit(TraceEvent('done', 'ok', 'installed', {}, round(time.monotonic() - t0, 3)))
        return [card], 'vector:x'
    monkeypatch.setattr(research, 'search_structured', search)
    monkeypatch.setattr(research, 'hub_model', lambda *args: {})
    monkeypatch.setattr(data, 'find_datasets', find)
    monkeypatch.setattr(data, 'fetch_sample', fetch)
    monkeypatch.setattr(data, 'build_set', build)
    monkeypatch.setattr(data, 'choose_prewired', lambda *args: w.prewired)
    monkeypatch.setattr(llm, 'ask_json', lambda *args, **kwargs: None)
    monkeypatch.setattr(models, 'available_models', lambda: [ModelInfo('fake', 'Fake model', ['text', 'image'], True)])
    monkeypatch.setattr(loop, 'run_fit', fit)
    w.wav = tmp_path / 'clip.wav'
    w.wav.write_bytes(b'RIFFxxxxWAVE')
    return w


def run(w, file=None, goal='recognise sounds', **kwargs):
    return run_auto('fake', file or w.wav, goal, w.events.append, **kwargs)


def steps(w):
    return list(dict.fromkeys(e.step for e in w.events))


def last(w, step, status=None):
    return next(e for e in reversed(w.events) if e.step == step and (status is None or e.status == status))


def test_native_modality(world, tmp_path):
    png = tmp_path / 'image'; png.write_bytes(b'\x89PNG\r\n\x1a\n')
    assert run(world, png) == ([], None)
    assert steps(world) == ['detect', 'done'] and not world.fit_calls


def test_unknown_format_refuses_with_citations(world, tmp_path):
    fasta = tmp_path / 'sequence'; fasta.write_text('>sequence\nACTG')
    assert run(world, fasta) == ([], None)
    done = last(world, 'done')
    assert done.data['needs_user'] and 'labelled examples' in done.message
    assert done.data['claims'][0]['source_url'].startswith('http')
    assert steps(world) == ['detect', 'research', 'done']


def test_user_folder_wins(world, tmp_path):
    folder = tmp_path / 'mine'
    run(world, examples=folder, goal='describe anything')
    assert world.build_calls[0]['folder'] == folder
    assert world.fit_calls[0].scope == 'closed'
    assert last(world, 'data', 'ok').data['origin'] == 'user'
    assert not world.fetch_calls


def test_fetched_dataset(world):
    world.candidates = [CAND]
    run(world)
    result = last(world, 'data', 'ok').data
    assert result['origin'] == 'fetched' and result['source'] == CAND.source_url
    assert result['licence'] == 'cc-by' and result['n_files'] == 120
    assert world.fit_calls[0].dataset == 'o_d'
    claim = next(c for c in result['claims'] if c['kind'] == 'dataset')
    assert claim['status'] == 'confirmed' and '1000' in claim['evidence']
    assert world.build_calls[0]['question'] == 'Which of these is in the audio clip?'


@pytest.mark.parametrize('failure', ['fetch', 'build'])
def test_candidate_failure_falls_back(world, failure):
    world.candidates = [CAND]
    setattr(world, failure + '_error', data.DataUnavailable('only 1 label had enough files'))
    run(world, goal='recognise music genres')
    assert any(e.step == 'data' and e.status == 'fail' and '1 label' in e.message for e in world.events)
    assert last(world, 'data', 'ok').data['origin'] == 'prewired'
    assert next(c for c in last(world, 'claims').data['claims'] if c['kind'] == 'dataset')['status'] == 'not_testable'


def test_research_down(world):
    world.research_error = research.ResearchUnavailable('network unreachable')
    assert run(world)[1] == 'vector:x'
    assert last(world, 'research', 'ok').data['fallback']


def test_no_data(world):
    world.prewired = None
    assert run(world) == ([], None)
    assert last(world, 'done').data['needs_user'] and not world.fit_calls


def test_two_nonrefuted_candidates_max(world):
    world.candidates = [data.DatasetCandidate(f'o/d{i}', 'Sounds', f'https://example.org/{i}') for i in range(4)]
    world.claim_status = {'o/d0': 'refuted'}
    world.fetch_error = data.DataUnavailable('unavailable')
    run(world)
    assert [c.hub_id for c, _, _ in world.fetch_calls] == ['o/d1', 'o/d2']


def test_probe_and_hook_order_and_clock(world):
    requests = []
    run(world, goal='describe anything', on_request=requests.append)
    assert requests == world.fit_calls and requests[0].scope == 'open'
    assert steps(world)[-3:] == ['verify', 'claims', 'done']
    assert last(world, 'claims').data['claims'][0]['status'] == 'confirmed'
    assert [e.t for e in world.events] == sorted(e.t for e in world.events)


def test_fake_run_does_not_confirm_live_specialist(world):
    world.origin = 'live'
    run(world)
    claims = last(world, 'claims').data['claims']
    assert all(c['status'] == 'not_testable' for c in claims)


@pytest.mark.parametrize('bad', ['model', 'missing', 'user'])
def test_errors_emit_one_pair(world, tmp_path, bad):
    if bad == 'user':
        world.build_error = data.DataUnavailable('bad folder')
    with pytest.raises((ValueError, data.DataUnavailable)):
        run_auto('unknown' if bad == 'model' else 'fake', tmp_path / 'missing' if bad == 'missing' else world.wav,
                 'recognise sounds', world.events.append, examples=tmp_path if bad == 'user' else None)
    assert len([e for e in world.events if e.step == 'error']) == 1
    assert world.events[-2].status == 'fail'


def test_fit_error_not_duplicated(world, monkeypatch):
    def fail(request, emit, *args, **kwargs):
        emit(TraceEvent('inspect', 'fail', 'bad model'))
        emit(TraceEvent('error', 'fail', 'stopped'))
        raise ValueError('bad model')
    monkeypatch.setattr(loop, 'run_fit', fail)
    with pytest.raises(ValueError): run(world)
    assert len([e for e in world.events if e.step == 'error']) == 1


@pytest.fixture
def fresh_jobs(monkeypatch):
    monkeypatch.setattr(jobs, '_jobs', {})
    monkeypatch.setattr(jobs, '_cancel_requested', set())
    monkeypatch.setattr(jobs, '_latest_id', None)
    monkeypatch.setattr(jobs, '_listeners', [])


def wait_for(job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = jobs.get(job_id)
        if job.status != 'running': return job
        time.sleep(.01)
    pytest.fail('job did not complete')


def test_start_auto_job_and_listeners(world, fresh_jobs):
    notifications = []; completed = threading.Event()
    def listen(*args):
        notifications.append(args)
        if args[1] is None: completed.set()
    jobs.add_listener(listen)
    job = wait_for(jobs.start_auto('fake', world.wav, 'recognise music genres'))
    assert completed.wait(2)
    assert job.status == 'done' and job.request.dataset == 'gtzan' and job.installed == 'vector:x'
    assert [event for _, event, _ in notifications if event] == job.events
    assert notifications[-1] == (job.id, None, job.cards)


@pytest.mark.parametrize('first_auto', [True, False])
def test_start_and_start_auto_share_singleflight(world, fresh_jobs, monkeypatch, first_auto):
    entered, release = threading.Event(), threading.Event()
    original = loop.run_fit
    def block(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        if 'on_cards' not in kwargs:
            return [], None
        return original(*args, **kwargs)
    monkeypatch.setattr(loop, 'run_fit', block)
    auto = lambda: jobs.start_auto('fake', world.wav, 'recognise sounds')
    manual = lambda: jobs.start(FitRequest('fake', 'gtzan'))
    first, second = (auto, manual) if first_auto else (manual, auto)
    job_id = first()
    try:
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match='already in progress'): second()
    finally:
        release.set(); wait_for(job_id)


def test_fake_mode_end_to_end_installs_and_cites(tmp_path, monkeypatch):
    from agent import planner as planner_mod
    from sense.specialists import FakeSpecialist, FAKE_CONCEPTS
    monkeypatch.setenv('SENSE_FAKE', '1')
    monkeypatch.setenv('SENSE_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr(planner_mod, '_preferred', None)
    clip = tmp_path / 'clip.wav'
    FakeSpecialist.write_sample(FAKE_CONCEPTS.index('rock'), clip, seed=1)
    events = []
    cards, installed = run_auto('fake', clip, 'Make it recognise music genres', events.append)
    assert installed == 'vector:gtzan'
    claims = next(e for e in reversed(events) if e.step == 'claims' and e.status == 'ok').data['claims']
    assert any(c['status'] == 'confirmed' and c['source_url'].startswith('http') and 'synthetic' in c['evidence'].lower() for c in claims)
    assert run_auto('fake', clip, 'Make it recognise music genres', [].append)[1] == installed


def test_dataset_search_unavailable_falls_back(world, monkeypatch):
    def unavailable(*args):
        raise research.ResearchUnavailable('malformed viewer response')
    monkeypatch.setattr(data, 'find_datasets', unavailable)
    assert run(world)[1] == 'vector:x'
    assert any(e.status == 'fail' and 'malformed viewer' in e.message for e in world.events)
    assert last(world, 'data', 'ok').data['origin'] == 'prewired'


def test_start_auto_failed_job_notifies(world, fresh_jobs, tmp_path):
    notifications = []; completed = threading.Event()
    def listen(*args):
        notifications.append(args)
        if args[1] is None: completed.set()
    jobs.add_listener(listen)
    job = wait_for(jobs.start_auto('fake', tmp_path / 'missing', 'recognise sounds'))
    assert completed.wait(2)
    assert job.status == 'failed' and job.error.startswith('ValueError:')
    assert notifications[-1] == (job.id, None, [])
    assert job.request == FitRequest('fake', '', '', 'closed')


def test_cli_refusal_and_failure_exit_codes(world, tmp_path, capsys):
    from agent.__main__ import main
    fasta = tmp_path / 'sequence'; fasta.write_text('>sequence\nACTG')
    args = ['--model', 'fake', '--goal', 'recognise sounds', '--file']
    assert main(args + [str(fasta)]) == 0
    assert 'Claim [unverified]' in capsys.readouterr().out
    assert main(args + [str(tmp_path / 'missing')]) == 1
    output = capsys.readouterr().out
    assert '[error] fail' in output and output.count('[error]') == 1


@pytest.mark.parametrize('automatic', [False, True])
def test_cancel_real_fit_and_auto_at_next_event(fresh_jobs, tmp_path, monkeypatch, automatic):
    from agent import planner as planner_mod
    monkeypatch.setenv('SENSE_FAKE', '1')
    monkeypatch.setenv('SENSE_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr(planner_mod, '_preferred', None)
    clip = tmp_path / 'clip.wav'; clip.write_bytes(b'RIFFxxxxWAVE')
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    notifications = []
    def listen(job_id, event, cards):
        notifications.append((job_id, event, cards))
        if event is not None and event.step == 'inspect' and event.status == 'start' and not entered.is_set():
            entered.set()
            assert release.wait(3)
        if event is None:
            assert jobs.get(job_id).error == 'Cancelled.'
            completed.set()
    jobs.add_listener(listen)
    assert jobs.cancel('unknown') is False
    job_id = jobs.start_auto('fake', clip, 'recognise music genres') if automatic else jobs.start(FitRequest('fake', 'gtzan'))
    try:
        assert entered.wait(2)
        assert jobs.cancel(job_id) is True
        assert jobs.cancel(job_id) is True
        assert jobs.get(job_id).status == 'running'
        with pytest.raises(RuntimeError, match='already in progress'):
            jobs.start(FitRequest('fake', 'gtzan'))
    finally:
        release.set()
    job = wait_for(job_id)
    assert completed.wait(2)
    assert job.status == 'failed' and job.error == 'Cancelled.'
    assert job.cards == [] and job.installed is None
    assert job.events[-1].step == 'error' and job.events[-1].status == 'fail'
    assert job.events[-1].message == 'Cancelled.'
    assert len([event for event in job.events if event.step == 'error']) == 1
    assert not any(event.step in ('install', 'done') for event in job.events)
    assert [event for _, event, _ in notifications if event is not None] == job.events
    assert [notice for notice in notifications if notice[1] is None] == [(job_id, None, [])]
    assert jobs.cancel(job_id) is False
    jobs.remove_listener(listen)
    following = wait_for(jobs.start(FitRequest('fake', 'gtzan')))
    assert following.status == 'done' and following.installed == 'vector:gtzan'
    assert jobs.cancel(following.id) is False


def test_cancel_before_return_without_another_event(world, fresh_jobs, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def finish(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return [], None
    monkeypatch.setattr(loop, 'run_fit', finish)
    job_id = jobs.start(FitRequest('fake', 'gtzan'))
    try:
        assert entered.wait(2)
        assert jobs.cancel(job_id)
    finally:
        release.set()
    job = wait_for(job_id)
    assert job.status == 'failed' and job.error == 'Cancelled.'
    assert len(job.events) == 1 and job.events[0].step == 'error'
