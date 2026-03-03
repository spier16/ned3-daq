import json

import nidaqmx
from nidaqmx.constants import (
    AcquisitionType, TerminalConfiguration, ExcitationSource, Coupling,
    TriggerType, Edge,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import threading
import queue
import os
import time
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
CHANNEL         = "cDAQ3Mod7/ai2"   # hydrophone AI channel
SAMPLE_RATE     = 12800              # Hz  (NI 9230 maximum)
BUFFER_SIZE     = 1280               # samples per read (100 ms chunks)
VOLTAGE_RANGE   = 5                  # ±5 V
IEPE_ENABLED    = False
PLOT_WINDOW     = 1.0                # seconds of data shown live
WRITE_DIR       = r"C:\Users\sapierso\Desktop"

# ── Counter / trigger config ───────────────────────────────────────────────────
COUNTER_OUT_CHAN = "cDAQ3/ctr0"      # pulse generator  → camera trigger
COUNTER_IN_CHAN  = "cDAQ3/ctr1"      # timestamp latching
TRIGGER_TERMINAL = "/cDAQ3/PFI0"     # BNC output → VEO 710 trigger input
PULSE_HIGH_TIME  = 1.0               # pulse width (s)

# ── Timebase / rollover constants ──────────────────────────────────────────────
TIMEBASE_HZ      = 80_000_000
TICKS_PER_SAMPLE = TIMEBASE_HZ // SAMPLE_RATE   # 6250 ticks per AI sample
TICKS_ROLLOVER   = 2**32
ROLLOVER_PERIOD_S = TICKS_ROLLOVER / TIMEBASE_HZ  # ≈ 53.69 s

# ── Shared state ───────────────────────────────────────────────────────────────
recording        = False
write_queue      = queue.Queue(maxsize=200)
plot_buffer      = np.zeros(int(SAMPLE_RATE * PLOT_WINDOW))
buffer_lock      = threading.Lock()
stop_event       = threading.Event()

# Set by acquisition_thread immediately after ai_task.start().
# Safe to read from any thread after timing_ready is set.
t_ai_start_perf  = None
t_ai_start_wall  = None

# Copy of t_ai_start_perf captured at the moment recording begins.
# Used ONLY for 32-bit rollover detection (needs ~1 s accuracy, not µs).
t_ci_start_perf  = None

timing_ready     = threading.Event()

# CI task lives for the duration of one recording session.
_ci_task         = None
_ci_task_lock    = threading.Lock()

# Per-recording trigger state.
trigger_used     = False
_trigger_lock    = threading.Lock()
trigger_result   = None   # dict written to _sync.json when recording stops

current_recording_file = None

# Signals for restarting the AI task at recording start.
_ai_restart_event   = threading.Event()   # main thread → acq thread: rebuild now
_ai_restarted_event = threading.Event()   # acq thread → main thread: new task live

# Sample accounting — all three variables are protected by _hw_samples_lock.
#
#   _ai_samples_acquired : total samples read from hardware since the last AI
#                          task restart, regardless of recording state.
#                          Reset to 0 at the start of each AI task.
#
#   _csv_start_offset    : hardware sample index of CSV row 0.  Set the first
#                          time a recording chunk is queued; None until then.
#                          Lets post-processing map a trigger to a CSV row:
#                            trigger_csv_row = fractional_sample_index
#                                           - csv_time0_hardware_sample_index
#
#   _hw_samples_written  : samples for which a write was attempted (including
#                          chunks dropped by a full queue).  Used as a second
#                          cross-check alongside fractional_sample_index.
_ai_samples_acquired = 0
_csv_start_offset    = None
_hw_samples_written  = 0
_hw_samples_lock     = threading.Lock()

# Live-plot trigger annotations.
trigger_events    = []
_trig_events_lock = threading.Lock()
trigger_vlines    = []


# ── Writer thread ──────────────────────────────────────────────────────────────
def writer_thread():
    """
    All disk I/O lives here. Queue message types:
        ("start", filepath, wall_datetime_str)
        ("data",  np.ndarray)
        ("stop",)
        ("quit",)
    Trigger sync data is written to a companion _sync.json, not inline here.
    """
    csv_file     = None
    sample_count = 0

    while True:
        item = write_queue.get()

        if item[0] == "start":
            _, filepath, wall_dt_str = item
            try:
                csv_file = open(filepath, "w")
                csv_file.write(
                    f"# hydrophone recording\n"
                    f"# file_start_utc  : {wall_dt_str}\n"
                    f"# sample_rate_hz  : {SAMPLE_RATE}\n"
                    f"# channel         : {CHANNEL}\n"
                    f"# trigger_terminal: {TRIGGER_TERMINAL}\n"
                    f"# sync_info       : see companion _sync.json\n"
                    f"#\n"
                    f"time_s,voltage_V\n"
                )
                sample_count = 0
                print(f"[writer] opened {filepath}")
            except OSError as e:
                print(f"ERROR: could not open recording file: {e}")
                csv_file = None

        elif item[0] == "data":
            if csv_file:
                try:
                    samples = item[1]
                    t0      = sample_count / SAMPLE_RATE
                    times   = t0 + np.arange(len(samples)) / SAMPLE_RATE
                    csv_file.write(
                        "\n".join(f"{t:.6f},{v:.6f}" for t, v in zip(times, samples))
                        + "\n"
                    )
                    sample_count += len(samples)   # only advance if write succeeded
                except OSError as e:
                    print(f"ERROR: write failed: {e}")

        elif item[0] == "stop":
            if csv_file:
                csv_file.close()
                csv_file = None

        elif item[0] == "quit":
            if csv_file:
                csv_file.close()
            break

        write_queue.task_done()


# ── CI task management ─────────────────────────────────────────────────────────
def _create_and_arm_ci_task():
    """
    Build the counter-input task that hardware-timestamps the trigger pulse.

    Signal flow:
      80 MHz timebase  →  ctr1 counts ticks continuously
      ctr0 output      →  ctr1 sample clock (latches current count on rising edge)
      AI StartTrigger  →  ctr1 arm-start trigger (counting begins with AI task)

    The arm-start trigger ensures tick 0 coincides with AI sample 0, giving
    sub-µs alignment between the timestamp and the audio sample stream.

    NOTE: if arm_start_trigger raises a DAQError on your firmware, comment out
    the three arm_start_trigger lines.  The CI task will then start ~100 µs
    before the AI task (software jitter), which is still far better than the
    original perf_counter approach and does not affect rollover detection.

    Returns the Task, already started (armed, waiting for AI StartTrigger).
    """
    ci = nidaqmx.Task("trigger_timestamp")

    ci_ch = ci.ci_channels.add_ci_count_edges_chan(
        COUNTER_IN_CHAN,
        edge=Edge.RISING,
    )
    # Count the 80 MHz onboard timebase ticks.
    ci_ch.ci_count_edges_term = "/cDAQ3/80MHzTimebase"

    # Latch the count on the rising edge of the CO pulse (one trigger per session).
    # CONTINUOUS mode: stopping the task early (at recording end) does not generate
    # DaqWarning 200010, which FINITE mode raises when samps_per_chan > triggers fired.
    # samps_per_chan=2: DAQmx requires a minimum buffer of 2 for clocked CI tasks.
    ci.timing.cfg_samp_clk_timing(
        rate=100,                               # nominal; real clock is ctr0 output
        source="/cDAQ3/Ctr0InternalOutput",     # internal route — no external cable
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=2,
    )

    # Arm: do not begin counting until the AI task fires its implicit start trigger,
    # so that CI tick 0 aligns with AI sample 0.
    ci.triggers.arm_start_trigger.trig_type    = TriggerType.DIGITAL_EDGE
    ci.triggers.arm_start_trigger.dig_edge_src = "/cDAQ3/ai/StartTrigger"

    ci.start()   # arms the task; counting begins when ai_task.start() fires
    return ci


def _write_sync_json(recording_filepath):
    """Write the companion _sync.json alongside the CSV."""
    sync_path = recording_filepath.replace(".csv", "_sync.json")
    payload = {
        "schema_version": 1,
        "recording": {
            "file":           os.path.basename(recording_filepath),
            "start_utc":      t_ai_start_wall.isoformat() if t_ai_start_wall else None,
            "sample_rate_hz": SAMPLE_RATE,
            "channel":        CHANNEL,
            # Hardware sample index of CSV row 0.  The trigger maps to CSV via:
            #   trigger_csv_row = fractional_sample_index
            #                   - csv_time0_hardware_sample_index
            "csv_time0_hardware_sample_index": _csv_start_offset,
        },
        "hardware": {
            "chassis":           "cDAQ3",
            "co_counter":        COUNTER_OUT_CHAN,
            "ci_counter":        COUNTER_IN_CHAN,
            "trigger_terminal":  TRIGGER_TERMINAL,
            "timebase_hz":       TIMEBASE_HZ,
            "ticks_per_ai_sample": TICKS_PER_SAMPLE,
        },
        "trigger": trigger_result if trigger_result is not None else {"fired": False},
    }
    with open(sync_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[sync] wrote {sync_path}")


# ── DAQ acquisition thread ─────────────────────────────────────────────────────
def acquisition_thread():
    """
    Owns the AI task for the lifetime of the application.  When the main thread
    sets _ai_restart_event, the current task is torn down and a new one is built.
    This restart is what lets the CI task arm-start in sync with the AI task at
    the beginning of each recording.
    """
    global t_ai_start_perf, t_ai_start_wall, plot_buffer
    global _hw_samples_written, _ai_samples_acquired, _csv_start_offset

    while not stop_event.is_set():
        _ai_restarted_event.clear()
        _ai_restart_event.clear()
        # Reset hardware sample counter for this AI task lifetime.
        with _hw_samples_lock:
            _ai_samples_acquired = 0

        with nidaqmx.Task() as ai_task:
            ch = ai_task.ai_channels.add_ai_voltage_chan(
                CHANNEL,
                terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                min_val=-VOLTAGE_RANGE,
                max_val=VOLTAGE_RANGE,
            )
            ch.ai_coupling = Coupling.AC
            if IEPE_ENABLED:
                ch.ai_excit_src = ExcitationSource.INTERNAL
                ch.ai_excit_val = 0.004

            ai_task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=BUFFER_SIZE * 4,
            )

            t_ai_start_wall = datetime.now()
            # This call fires /cDAQ3/ai/StartTrigger, which releases the CI task
            # from its arm-start state so both begin counting from the same edge.
            ai_task.start()
            t_ai_start_perf = time.perf_counter()
            timing_ready.set()
            _ai_restarted_event.set()   # unblocks toggle_recording()

            print(f"[acq] task live | start: {t_ai_start_wall.isoformat()}")

            while not stop_event.is_set() and not _ai_restart_event.is_set():
                try:
                    samples = np.array(
                        ai_task.read(
                            number_of_samples_per_channel=BUFFER_SIZE,
                            timeout=0.5,
                        )
                    )
                except nidaqmx.errors.DaqReadError:
                    continue

                with buffer_lock:
                    plot_buffer = np.roll(plot_buffer, -len(samples))
                    plot_buffer[-len(samples):] = samples

                # Capture recording flag once; GIL makes bool reads atomic.
                currently_recording = recording
                with _hw_samples_lock:
                    _ai_samples_acquired += len(samples)
                    if currently_recording:
                        _hw_samples_written += len(samples)
                        if _csv_start_offset is None:
                            # First chunk of this recording — mark CSV row 0.
                            _csv_start_offset = _ai_samples_acquired - len(samples)

                if currently_recording:
                    try:
                        write_queue.put_nowait(("data", samples))
                    except queue.Full:
                        print("WARNING: write queue full — samples dropped.")

            timing_ready.clear()
            if _ai_restart_event.is_set():
                print("[acq] restarting AI task for recording sync...")
            else:
                print("[acq] stopped.")


# ── Trigger firing ─────────────────────────────────────────────────────────────
def fire_trigger(_event=None):
    """Button callback. Checks guards then spawns background thread."""
    if not recording:
        print("[trigger] not recording — blocked.")
        return

    with _trigger_lock:
        if trigger_used:
            print("[trigger] already fired once this recording.")
            return

    # Update UI immediately (safe: called from matplotlib event thread).
    btn_trigger.color      = "lightgray"
    btn_trigger.hovercolor = "lightgray"
    btn_trigger.label.set_text("✓ Triggered")
    btn_trigger.set_active(False)
    fig.canvas.draw_idle()

    threading.Thread(target=_do_fire_trigger, daemon=True).start()


def _do_fire_trigger():
    global trigger_used, trigger_result

    # Double-check inside the lock (race guard for rapid double-click).
    with _trigger_lock:
        if trigger_used:
            return
        trigger_used = True

    # ── Fire the TTL pulse on ctr0 / PFI0 ──────────────────────────────────────
    with nidaqmx.Task() as co_task:
        co_task.co_channels.add_co_pulse_chan_time(
            COUNTER_OUT_CHAN,
            low_time=1e-6,
            high_time=PULSE_HIGH_TIME,
        )
        co_task.timing.cfg_implicit_timing(
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=1,
        )
        co_task.export_signals.export_signal(
            nidaqmx.constants.Signal.COUNTER_OUTPUT_EVENT,
            TRIGGER_TERMINAL,
        )
        # Capture perf_counter BEFORE start() so the rollover window check uses
        # the time just before the rising edge, not 1 s later after
        # wait_until_done().  If captured after, a pulse fired within
        # PULSE_HIGH_TIME of a rollover boundary would yield rollover_count+1.
        t_before_pulse = time.perf_counter()
        co_task.start()
        # The rising edge of this pulse is what latched ctr1's count in hardware.
        # By the time wait_until_done() returns, the latch has been complete for
        # PULSE_HIGH_TIME seconds — the read below is instantaneous.
        co_task.wait_until_done(timeout=max(PULSE_HIGH_TIME * 2, 1.0))

    # ── Read the hardware-latched 80 MHz tick count ─────────────────────────────
    with _ci_task_lock:
        ci = _ci_task
        if ci is None:
            print("[trigger] ERROR: CI task unavailable — no sync data recorded.")
            return
        raw = ci.read(number_of_samples_per_channel=1, timeout=2.0)

    latched_value = raw if isinstance(raw, int) else int(raw[0])

    # Snapshot the software sample counter immediately after reading the latch.
    # Used to cross-check fractional_sample_index: if they differ by more than
    # one buffer's worth (~1280), samples were dropped by the write queue.
    with _hw_samples_lock:
        hw_samples_at_trigger = _hw_samples_written

    # ── 32-bit rollover compensation ────────────────────────────────────────────
    # t_before_pulse is at most a few ms before the actual rising edge — safely
    # within the ±26 s margin around any 53.69 s rollover boundary.
    elapsed_s      = t_before_pulse - t_ci_start_perf
    rollover_count = int(elapsed_s / ROLLOVER_PERIOD_S)
    true_ticks     = latched_value + rollover_count * TICKS_ROLLOVER

    t_rel_s        = true_ticks / TIMEBASE_HZ
    frac_sample    = true_ticks / TICKS_PER_SAMPLE
    nearest_sample = round(frac_sample)
    subsample_us   = (frac_sample - nearest_sample) / SAMPLE_RATE * 1e6

    trigger_result = {
        "fired":                        True,
        "true_tick_count":              int(true_ticks),
        "rollover_count":               rollover_count,
        "time_since_recording_start_s": t_rel_s,
        "fractional_sample_index":      frac_sample,
        "nearest_sample_index":         nearest_sample,
        "subsample_offset_us":          subsample_us,
        # Cross-check: (fractional_sample_index - hardware_samples_read_at_trigger)
        # should equal csv_time0_hardware_sample_index within ±BUFFER_SIZE.
        # A larger residual means samples were discarded at recording start
        # (recording flag was set after the first chunk completed — rare).
        # NOTE: write_queue drops cannot be detected here; both this counter and
        # fractional_sample_index are unaffected by queue drops.
        "hardware_samples_read_at_trigger": hw_samples_at_trigger,
    }

    ev = {"t_relative_s": t_rel_s, "sample_index": frac_sample}
    with _trig_events_lock:
        trigger_events.append(ev)

    print(
        f"[trigger] FIRED | "
        f"t = {t_rel_s * 1e3:.3f} ms into recording | "
        f"sample ≈ {frac_sample:.3f} | "
        f"sub-sample offset = {subsample_us:+.2f} µs"
    )


# ── Recording toggle ───────────────────────────────────────────────────────────
def toggle_recording(event):
    global recording, current_recording_file, _ci_task
    global t_ci_start_perf, trigger_used, trigger_result
    global _hw_samples_written, _csv_start_offset

    if not recording:
        # ── Start recording ────────────────────────────────────────────────────
        if not timing_ready.is_set():
            print("WARNING: acquisition not ready yet.")
            return

        btn_record.set_active(False)    # prevent double-click during restart

        # 1. Build and arm CI task (waits for the AI start trigger).
        ci = _create_and_arm_ci_task()
        with _ci_task_lock:
            _ci_task = ci

        # 2. Signal acquisition thread to restart the AI task.
        #    When ai_task.start() fires inside the thread, it simultaneously
        #    releases the CI arm-start, so both tasks begin on the same clock edge.
        timing_ready.clear()
        _ai_restart_event.set()
        _ai_restarted_event.wait()      # blocks until new AI task is live (~0.5 s)

        # t_ai_start_perf was just set by the acquisition thread; safe to read.
        t_ci_start_perf = t_ai_start_perf

        # 3. Reset per-recording state.
        # _ai_samples_acquired is reset by acquisition_thread on AI task restart.
        with _hw_samples_lock:
            _hw_samples_written = 0
            _csv_start_offset   = None
        trigger_used   = False
        trigger_result = None
        with _trig_events_lock:
            trigger_events.clear()
        for vl in trigger_vlines:
            try:
                vl.remove()
            except Exception:
                pass
        trigger_vlines.clear()

        # 4. Open CSV.
        current_recording_file = os.path.join(
            WRITE_DIR, datetime.now().strftime("hydrophone_%Y%m%d_%H%M%S.csv")
        )
        write_queue.put(("start", current_recording_file, t_ai_start_wall.isoformat()))
        recording = True

        btn_record.set_active(True)
        btn_record.label.set_text("■ Stop Recording")
        btn_record.color      = "tomato"
        btn_trigger.set_active(True)
        btn_trigger.color      = "gold"
        btn_trigger.hovercolor = "lightyellow"
        btn_trigger.label.set_text("⚡ Fire Trigger")
        print(f"[ui] recording → {current_recording_file}")

    else:
        # ── Stop recording ─────────────────────────────────────────────────────
        recording = False
        write_queue.put(("stop",))

        # Write sync sidecar before closing CI task.
        if current_recording_file:
            _write_sync_json(current_recording_file)

        # Stop and close CI task.
        with _ci_task_lock:
            if _ci_task is not None:
                try:
                    _ci_task.stop()
                    _ci_task.close()
                except nidaqmx.errors.DaqError as e:
                    print(f"[ci] warning during close: {e}")
                _ci_task = None

        btn_record.label.set_text("● Enable Recording")
        btn_record.color      = "limegreen"
        btn_trigger.set_active(False)
        btn_trigger.color      = "lightgray"
        btn_trigger.hovercolor = "lightgray"
        btn_trigger.label.set_text("⚡ Fire Trigger")
        print("[ui] recording stopped.")

    fig.canvas.draw_idle()


# ── UI / plotting ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(bottom=0.22)

time_axis      = np.linspace(-PLOT_WINDOW, 0, int(SAMPLE_RATE * PLOT_WINDOW))
(line,)        = ax.plot(time_axis, plot_buffer, lw=0.8, color="dodgerblue")

ax.set_xlim(-PLOT_WINDOW, 0)
ax.set_ylim(-VOLTAGE_RANGE, VOLTAGE_RANGE)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (V)")
ax.set_title("Hydrophone — Live Waveform")
ax.grid(True, alpha=0.3)

_plot_start_time    = time.time()
_last_trigger_count = 0


def animate(_):
    global _last_trigger_count

    with buffer_lock:
        data = plot_buffer.copy()

    line.set_ydata(data)

    peak   = np.max(np.abs(data))
    margin = peak * 0.2 or 0.001
    ax.set_ylim(-peak - margin, peak + margin)

    elapsed = time.time() - _plot_start_time
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    ax.set_title(f"Hydrophone — Live Waveform    {h:02d}:{m:02d}:{s:02d}")

    with _trig_events_lock:
        new_events          = trigger_events[_last_trigger_count:]
        _last_trigger_count = len(trigger_events)

    now_perf = time.perf_counter()
    for ev in new_events:
        if t_ai_start_perf is None:
            continue
        t_fired_perf = t_ai_start_perf + ev["t_relative_s"]
        x_on_plot    = -(now_perf - t_fired_perf)
        if -PLOT_WINDOW <= x_on_plot <= 0:
            vl = ax.axvline(x_on_plot, color="crimson", lw=1.2, ls="--", alpha=0.8)
            ax.text(
                x_on_plot, ax.get_ylim()[1] * 0.9,
                f" T+{ev['t_relative_s']*1e3:.1f} ms\n ≈ s{ev['sample_index']:.0f}",
                color="crimson", fontsize=7, va="top",
            )
            trigger_vlines.append(vl)

    return (line, *trigger_vlines)


# ── Buttons ────────────────────────────────────────────────────────────────────
ax_btn_rec  = plt.axes([0.25, 0.05, 0.25, 0.08])
btn_record  = Button(ax_btn_rec,  "● Enable Recording", color="limegreen")
btn_record.on_clicked(toggle_recording)

ax_btn_trig = plt.axes([0.55, 0.05, 0.22, 0.08])
btn_trigger = Button(ax_btn_trig, "⚡ Fire Trigger",    color="lightgray")
btn_trigger.hovercolor = "lightgray"
btn_trigger.set_active(False)   # enabled only while a recording is active
btn_trigger.on_clicked(fire_trigger)

# ── Start everything ───────────────────────────────────────────────────────────
daq_thread = threading.Thread(target=acquisition_thread, daemon=True)
daq_thread.start()

csv_thread = threading.Thread(target=writer_thread, daemon=True)
csv_thread.start()

ani = animation.FuncAnimation(
    fig, animate, interval=50, blit=False, cache_frame_data=False
)

try:
    plt.show()
finally:
    recording = False
    stop_event.set()
    _ai_restart_event.set()     # unblock acq thread if it is mid-restart wait
    daq_thread.join(timeout=3)
    with _ci_task_lock:
        if _ci_task is not None:
            try:
                _ci_task.stop()
                _ci_task.close()
            except Exception:
                pass
    write_queue.put(("quit",))
    csv_thread.join(timeout=30)
    if csv_thread.is_alive():
        print("WARNING: writer thread did not finish cleanly — output may be truncated.")
    print("Done.")
