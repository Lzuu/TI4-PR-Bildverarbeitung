"""Detailed suit timing is per frame and must not change rendered pixels."""
import csv
import logging
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import main as m
import perf
from suit_rig import SuitRenderer
from loop_harness import run_loop,FakeCamera,green_frame
from test_main_loop import CountingIdentifier
from tools.preview_suit import scene

SUBSTAGES=('suit_masks','suit_fabric','suit_rig','suit_sprites','suit_helmet','suit_restore')


def test_repeated_substages_are_summed_before_one_sample_per_frame():
    clock=[0.];stats=perf.FrameStats(interval=0,clock=lambda:clock[0],log=None)
    stats.start_frame()
    with perf.StageAccumulator(stats,clock=lambda:clock[0]) as timing:
        with timing.stage('sprites'):clock[0]+=.01
        with timing.stage('helmet'):clock[0]+=.02
        with timing.stage('sprites'):clock[0]+=.03
    result=stats.end_frame()['stages']
    assert result['sprites']['count']==1
    assert result['sprites']['median']==pytest.approx(40.)
    assert result['helmet']['median']==pytest.approx(20.)


def test_optional_timer_does_not_read_clock_and_keeps_exceptions():
    def forbidden():raise AssertionError('clock without stats')
    with pytest.raises(ValueError,match='original'):
        with perf.StageAccumulator(None,clock=forbidden) as timing:
            with timing.stage('sprites'):raise ValueError('original')


def test_active_timer_flushes_partial_work_without_swallowing_errors():
    clock=[0.];stats=perf.FrameStats(interval=0,clock=lambda:clock[0],log=None)
    stats.start_frame()
    with pytest.raises(ValueError,match='render failure'):
        with perf.StageAccumulator(stats,clock=lambda:clock[0]) as timing:
            with timing.stage('sprites'):
                clock[0]=.01
                raise ValueError('render failure')
    assert stats.end_frame()['stages']['sprites']['median']==pytest.approx(10.)


@pytest.mark.parametrize('kind',['down','cross','up'])
def test_recording_does_not_change_compositing_and_counts_each_group_once(kind):
    raw,mask,pose,face,keep=scene(kind)
    expected=SuitRenderer().render(raw.copy(),raw,mask,pose,face,keep,now=0)
    stats=perf.FrameStats(interval=0,log=None);stats.start_frame()
    actual=SuitRenderer().render(raw.copy(),raw,mask,pose,face,keep,now=0,stats=stats)
    stages=stats.end_frame()['stages']
    assert np.array_equal(expected,actual)
    assert set(SUBSTAGES)<=set(stages)
    assert all(stages[name]['count']==1 for name in SUBSTAGES)


def test_missing_player_does_not_claim_render_stages_ran():
    raw,mask,pose,face,keep=scene()
    stats=perf.FrameStats(interval=0,log=None);stats.start_frame()
    actual=SuitRenderer().render(raw.copy(),raw,np.zeros_like(mask),now=0,stats=stats)
    stages=stats.end_frame()['stages']
    assert np.array_equal(actual,raw)
    assert set(stages)=={'suit_masks'}


def test_untracked_player_only_records_real_mask_rig_and_restore_work():
    raw,mask,pose,face,keep=scene()
    stats=perf.FrameStats(interval=0,log=None);stats.start_frame()
    actual=SuitRenderer().render(raw.copy(),raw,mask,keep=keep,now=0,stats=stats)
    stages=stats.end_frame()['stages']
    assert np.array_equal(actual,raw)
    assert set(stages)=={'suit_masks','suit_rig','suit_restore'}


def test_real_loop_logs_and_exports_detailed_suit_times(monkeypatch,tmp_path,caplog):
    caplog.set_level(logging.INFO)
    path=tmp_path/'perf.csv'
    run_loop(monkeypatch,tmp_path,FakeCamera([green_frame()]),3,
             identifier=CountingIdentifier(),BG_MODE='greenscreen',SUIT_ENABLED=True,
             PERF_LOG_INTERVAL=0.,PERF_CSV=str(path))
    lines=[r.getMessage() for r in caplog.records if r.getMessage().startswith('Perf ')]
    for name in SUBSTAGES:assert any(name in line for line in lines)
    with path.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    assert set(SUBSTAGES)<={row['stage'] for row in rows}
    assert all(int(row['count'])<=int(row['frames']) for row in rows if row['stage'] in SUBSTAGES)
    frame={}
    for row in rows+[{'stage':'frame'}]:
        if row['stage']=='frame':
            if 'suit' in frame:
                assert sum(frame.get(name,0.) for name in SUBSTAGES)<=frame['suit']+.1
            frame={}
        elif 'median_ms' in row:
            frame[row['stage']]=float(row['median_ms'])


def test_detailed_suit_rows_fit_in_the_actual_overlay(monkeypatch):
    stats=perf.FrameStats(interval=0,log=None);stats.start_frame()
    for i in range(18):stats.add('expensive_'+str(i),1.)
    stats.add('suit',.02)
    for name in SUBSTAGES:stats.add(name,.001)
    stats.end_frame()
    drawn=[]
    monkeypatch.setattr(m,'draw_text',lambda frame,text,*args,**kwargs:drawn.append(text))
    m.draw_perf_overlay(np.zeros((540,960,3),np.uint8),stats.overlay_lines())
    for name in SUBSTAGES:assert any(name+':' in line for line in drawn)
    assert len(drawn)<=12


def test_help_has_no_dead_suit_flag_and_old_flag_is_rejected():
    program=Path(m.__file__)
    result=subprocess.run([sys.executable,str(program),'--help'],capture_output=True,text=True,timeout=15)
    assert result.returncode==0 and '--suit-diagnostics' not in result.stdout
    result=subprocess.run([sys.executable,str(program),'--suit-diagnostics'],capture_output=True,text=True,timeout=15)
    assert result.returncode==2 and 'unrecognized arguments' in result.stderr
