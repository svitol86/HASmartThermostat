"""Unit tests for the PID controller: set point recovery and anti-windup with external compensation."""

import pytest

from custom_components.smart_thermostat.pid_controller import PID


def run(pid, temps, sp, ext=None, t0=1000.0, dt=60.0):
    """Feed a list of temperatures at a fixed rate and return the effective outputs (PID + external)."""
    out = []
    t = t0
    last = None
    for temp in temps:
        raw, _ = pid.calc(temp, sp, t, last, ext)
        out.append(raw + pid.external)
        last = t
        t += dt
    return out


def make(**kw):
    args = dict(kp=50, ki=0.005, kd=0, ke=1.3, out_min=0, out_max=100, sampling_period=60)
    args.update(kw)
    return PID(**args)


def test_recovery_disabled_by_default_keeps_behaviour():
    pid = make()
    assert pid.recovery_offset == 0
    run(pid, [20.0] * 3, 21.0, ext=10.0)
    assert not pid.recovering
    # error 1 K, ki 0.005/s, 2 integration steps of 60 s
    assert pid.integral == pytest.approx(0.6)


def test_recovery_holds_output_and_freezes_integral():
    pid = make(recovery_offset=0.3)
    pid.integral = 10.0
    run(pid, [18.0], 17.0, ext=10.0)              # first call: nothing to recover
    assert not pid.recovering
    out = run(pid, [18.0, 18.5, 19.0, 20.0, 20.6], 21.0, ext=10.0, t0=2000.0)
    assert pid.recovering
    assert all(o == pytest.approx(100.0) for o in out)
    assert pid.integral == pytest.approx(10.0)
    # P + I + D + E still add up to the held output
    assert pid.proportional + pid.integral + pid.derivative + pid.external == pytest.approx(100.0)


def test_recovery_ends_inside_band_and_pid_resumes():
    pid = make(recovery_offset=0.3)
    run(pid, [18.0], 17.0, ext=10.0)
    run(pid, [19.0, 20.0], 21.0, ext=10.0, t0=2000.0)
    assert pid.recovering
    out = run(pid, [20.8], 21.0, ext=10.0, t0=3000.0)
    assert not pid.recovering
    # normal PID: 50 * 0.2 + I + 1.3 * 11
    assert out[0] == pytest.approx(10.0 + pid.integral + 14.3)


def test_recovery_cancelled_by_set_point_drop():
    pid = make(recovery_offset=0.3)
    run(pid, [18.0], 17.0, ext=10.0)
    run(pid, [18.0], 21.0, ext=10.0, t0=2000.0)
    assert pid.recovering
    out = run(pid, [18.1], 17.0, ext=10.0, t0=3000.0)
    assert not pid.recovering
    assert out[0] < 100


def test_no_recovery_when_already_inside_band():
    pid = make(recovery_offset=0.3)
    run(pid, [20.0], 20.0, ext=10.0)
    run(pid, [20.8], 21.0, ext=10.0, t0=2000.0)
    assert not pid.recovering


def test_recovery_resumes_after_restart_far_below():
    pid = make(recovery_offset=0.3)
    out = run(pid, [19.0, 19.1], 21.0, ext=10.0)
    assert pid.recovering
    assert out[-1] == pytest.approx(100.0)


def test_cooling_recovery_drives_minimum():
    pid = make(recovery_offset=0.3, out_min=-100, out_max=0, ke=0)
    run(pid, [24.0], 26.0)
    out = run(pid, [26.0, 25.5], 23.0, t0=2000.0)
    assert pid.recovering
    assert all(o == pytest.approx(-100.0) for o in out)


def test_heat_mode_never_starts_cooling_recovery():
    pid = make(recovery_offset=0.3)                # out_min 0: heating only
    run(pid, [21.0], 21.0, ext=10.0)
    run(pid, [21.0], 17.0, ext=10.0, t0=2000.0)
    assert not pid.recovering


def test_anti_windup_accounts_for_external_compensation():
    """Output saturated only thanks to the external term: the integral must not grow."""
    pid = make(kp=50, ki=0.005, ke=1.3)
    pid.integral = 49.0
    # error 0.9 K: P 45 + I 49 = 94 (< 100), E = 1.3 * 11 = 14.3 -> effective 108 -> saturated
    run(pid, [20.1] * 10, 21.0, ext=10.0)
    assert pid.integral == pytest.approx(49.0)     # first call has dt = 0, then saturated


def test_anti_windup_without_external_is_unchanged():
    pid = make(kp=50, ki=0.005, ke=0)
    pid.integral = 49.0
    run(pid, [20.1] * 10, 21.0)
    assert pid.integral == pytest.approx(49.0 + 9 * 0.005 * 0.9 * 60)


def test_clear_samples_ends_recovery():
    pid = make(recovery_offset=0.3)
    run(pid, [18.0], 21.0, ext=10.0)
    assert pid.recovering
    pid.clear_samples()
    assert not pid.recovering
