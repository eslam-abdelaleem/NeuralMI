# tests/test_device.py
"""The device a call trains on when it names none (neural_mi/device.py).

The decision is tested with injected step times, so it does not depend on the
machine; one end-to-end call checks the timing path and the recorded device.
"""
import math

import numpy as np
import pytest

import neural_mi as nmi
from neural_mi import device as dev
from neural_mi.analysis.task import run_training_task
from neural_mi.validation import ParameterValidator


def _tasks(n, n_epochs=100):
    return [(None, None, {'n_epochs': n_epochs}, f'run_{i}') for i in range(n)]


@pytest.fixture
def timings(monkeypatch):
    """Step times by device, as (seconds per step, steps per epoch). A device left out fails the test if timed."""
    monkeypatch.setattr(dev, 'accelerator', lambda: 'mps')
    times = {}

    def fake(task, device, threads):
        assert device in times, f"{device} was timed"
        return times[device]
    monkeypatch.setattr(dev, '_time', fake)
    return times


def test_a_named_device_is_kept(timings):
    tasks = [(None, None, {'device': 'cpu', 'n_epochs': 100}, 'run_0')]
    assert dev.with_device(tasks, 1) is tasks


def test_without_a_gpu_nothing_is_timed(timings, monkeypatch):
    monkeypatch.setattr(dev, 'accelerator', lambda: None)
    assert dev.choose_device(_tasks(1), 1) == 'cpu'


def test_a_worker_never_times(timings, monkeypatch):
    monkeypatch.setattr(dev.mp, 'parent_process', lambda: object())
    assert dev.choose_device(_tasks(1), 1) == 'cpu'


def test_a_short_call_does_not_time_the_gpu(timings):
    timings['cpu'] = (1e-3, 100)                    # 10 s of training in all
    assert dev.choose_device(_tasks(1), 1) == 'cpu'


def test_the_gpu_is_chosen_when_clearly_faster(timings):
    timings.update(cpu=(1e-2, 1000), mps=(2e-3, 1000))
    assert dev.choose_device(_tasks(1), 1) == 'mps'


def test_parallel_workers_need_the_gpu_to_beat_them_together(timings):
    # Five times faster than one worker, half as fast as ten.
    timings.update(cpu=(1e-2, 1000), mps=(2e-3, 1000))
    assert dev.choose_device(_tasks(10), 10) == 'cpu'


def test_a_small_gain_is_not_worth_the_gpu(timings):
    timings.update(cpu=(1e-2, 1000), mps=(1e-2 / 1.2, 1000))
    assert dev.choose_device(_tasks(1), 1) == 'cpu'


def test_timing_a_task_does_not_train_it():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 3)).astype(np.float32)
    params = {'device': 'cpu', 'batch_size': 64, '_time_steps': True}
    ParameterValidator({'base_params': params}).apply_defaults()
    timed = run_training_task((x, x + rng.standard_normal(x.shape).astype(np.float32), params, 'run_0'))
    assert set(timed) == {'seconds_per_step', 'steps_per_epoch'}
    assert timed['seconds_per_step'] > 0
    assert timed['steps_per_epoch'] == math.ceil(int(1000 * params['train_fraction']) / 64)


def test_a_call_that_names_no_device_records_the_one_it_used():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 3))
    result = nmi.run(x, x + rng.standard_normal(x.shape), mode='estimate', training=nmi.Training(n_epochs=3),
                     seed=0, show_progress=False, verbose=False)
    assert result.runs['device'].tolist() == ['cpu']      # too short a call to time a GPU
