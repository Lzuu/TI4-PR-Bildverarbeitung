"""Diagnostics must not create a log line or unbounded history per frame."""
import logging
import frame_metrics as metrics


def test_disabled_diagnostics_store_nothing():
    meter=metrics.FrameMetrics(False)
    with meter.measure('pose'):
        pass
    meter.finish(0)
    assert not meter.samples


def test_enabled_diagnostics_log_one_aggregated_window(monkeypatch,caplog):
    clock=[0.]
    monkeypatch.setattr(metrics.time,'monotonic',lambda:clock[0])
    meter=metrics.FrameMetrics(True,interval=5)
    with caplog.at_level(logging.INFO):
        for tick in range(6):
            clock[0]=float(tick)
            with meter.measure('pose'):
                pass
            meter.finish(metrics.time.perf_counter())
    entries=[record for record in caplog.records if record.name=='frame_metrics']
    assert len(entries)==1
    assert 'p95_ms' in entries[0].message and 'frames/s' in entries[0].message
