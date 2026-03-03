import nidaqmx
from nidaqmx.constants import (
    AcquisitionType, TerminalConfiguration, ExcitationSource, Coupling
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

# ── Configuration ─────────────────────────────────────────────────────────────
CHANNEL         = "cDAQ3Mod7/ai2"   # hydrophone AI channel
SAMPLE_RATE     = 12800              # Hz
BUFFER_SIZE     = 1280               # samples per read (100 ms chunks)
VOLTAGE_RANGE   = 5                  # ±5 V
IEPE_ENABLED    = False
PLOT_WINDOW     = 1.0                # seconds of data shown live
WRITE_DIR       = r"C:\Users\sapierso\Desktop"

# ── Trigger config ────────────────────────────────────────────────────────────
COUNTER_OUT_CHAN = "cDAQ3/ctr0"
TRIGGER_TERMINAL = "/cDAQ3/PFI0"    # BNC jack → VEO 710 trigger input
PULSE_HIGH_TIME  = 1.0            # pulse width (seconds)

# ── Shared state ──────────────────────────────────────────────────────────────
recording       = False
write_queue     = queue.Queue(maxsize=200)
plot_buffer     = np.zeros(int(SAMPLE_RATE * PLOT_WINDOW))
buffer_lock     = threading.Lock()
stop_event      = threading.Event()

t_ai_start_perf       = None  # time.perf_counter() at AI task.start()
t_ai_start_wall       = None  # datetime.now()       at AI task.start()
t_recording_start_perf = None  # time.perf_counter() at recording start
timing_ready          = threading.Event()

trigger_events  = []    # list of dicts, appended by fire_trigger()
trigger_lock    = threading.Lock()


# ── Writer thread ─────────────────────────────────────────────────────────────
def writer_thread():
    """
    All disk I/O lives here. Queue message types:
        ("start",   filepath, wall_datetime_str)
        ("data",    np.ndarray)
        ("trigger", dict)
        ("stop",)
        ("quit",)
    """
    csv_file     = None
    sample_count = 0

    while True:
        item = write_queue.get()

        if item[0] == "start":
            _, filepath, wall_dt_str = item
            try:
                csv_file     = open(filepath, "w")
                csv_file.write(
                    f"# hydrophone recording\n"
                    f"# file_start_utc  : {wall_dt_str}\n"
                    f"# sample_rate_hz  : {SAMPLE_RATE}\n"
                    f"# channel         : {CHANNEL}\n"
                    f"# trigger_terminal: {TRIGGER_TERMINAL}\n"
                    f"#\n"
                    f"# TRIGGER EVENTS appear as comment lines mid-file:\n"
                    f"#   # TRIGGER  t_rel_s=<s>  sample_idx=<fractional>\n"
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
                    sample_count += len(samples)
                    csv_file.write(
                        "\n".join(f"{t:.6f},{v:.6f}" for t, v in zip(times, samples))
                        + "\n"
                    )
                except OSError as e:
                    print(f"ERROR: write failed: {e}")

        elif item[0] == "trigger":
            ev = item[1]
            line = (
                f"# TRIGGER"
                f"  t_rel_s={ev['t_relative_s']:.9f}"
                f"  sample_idx={ev['sample_index']:.3f}\n"
            )
            if csv_file:
                try:
                    csv_file.write(line)
                    csv_file.flush()
                except OSError as e:
                    print(f"ERROR: could not write trigger annotation: {e}")
            print(
                f"[writer] TRIGGER {'logged' if csv_file else '(not recording)'}  →  "
                f"t={ev['t_relative_s']*1e3:.3f} ms  |  sample ≈ {ev['sample_index']:.1f}"
            )

        elif item[0] == "stop":
            if csv_file:
                csv_file.close()
                csv_file = None

        elif item[0] == "quit":
            if csv_file:
                csv_file.close()
            break

        write_queue.task_done()


# ── DAQ acquisition thread ────────────────────────────────────────────────────
def acquisition_thread():
    global plot_buffer, t_ai_start_perf, t_ai_start_wall

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
        ai_task.start()
        t_ai_start_perf = time.perf_counter()
        timing_ready.set()

        print(f"[acq] task live  |  AI start: {t_ai_start_wall.isoformat()}")

        while not stop_event.is_set():
            samples = np.array(
                ai_task.read(number_of_samples_per_channel=BUFFER_SIZE, timeout=2.0)
            )

            with buffer_lock:
                plot_buffer = np.roll(plot_buffer, -len(samples))
                plot_buffer[-len(samples):] = samples

            if recording:
                try:
                    write_queue.put_nowait(("data", samples))
                except queue.Full:
                    print("WARNING: write queue full — samples dropped.")

        print("[acq] stopped.")


# ── Trigger firing ────────────────────────────────────────────────────────────
def fire_trigger(_event=None):
    """Spawns a background thread so the UI is never blocked by the pulse."""
    threading.Thread(target=_do_fire_trigger, daemon=True).start()


def _do_fire_trigger():
    """
    Fires a hardware-timed TTL pulse on PFI0 → VEO 710, then records the
    perf_counter timestamp so we know where in the hydrophone stream it fell.
    """
    if not timing_ready.is_set():
        print("[trigger] not ready — acquisition hasn't started yet.")
        return

    if t_recording_start_perf is None:
        print("[trigger] WARNING: no active recording — trigger fired but t_rel_s will not be meaningful.")

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

        co_task.start()
        t_fire_perf = time.perf_counter()
        co_task.wait_until_done(timeout=max(PULSE_HIGH_TIME * 2, 1.0))

    t_relative_s = t_fire_perf - t_recording_start_perf
    sample_index = t_relative_s * SAMPLE_RATE

    ev = {"t_relative_s": t_relative_s, "sample_index": sample_index}

    with trigger_lock:
        trigger_events.append(ev)

    write_queue.put(("trigger", ev))

    print(
        f"[trigger] FIRED  |  "
        f"t={t_relative_s*1e3:.3f} ms  |  sample ≈ {sample_index:.1f}"
    )


# ── UI / plotting ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(bottom=0.22)

time_axis      = np.linspace(-PLOT_WINDOW, 0, int(SAMPLE_RATE * PLOT_WINDOW))
(line,)        = ax.plot(time_axis, plot_buffer, lw=0.8, color="dodgerblue")
trigger_vlines = []

ax.set_xlim(-PLOT_WINDOW, 0)
ax.set_ylim(-VOLTAGE_RANGE, VOLTAGE_RANGE)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (V)")
ax.set_title("Hydrophone — Live Waveform")
ax.grid(True, alpha=0.3)

plot_start_time    = time.time()
last_trigger_count = 0


def animate(_frame):
    global last_trigger_count

    with buffer_lock:
        data = plot_buffer.copy()

    line.set_ydata(data)

    peak   = np.max(np.abs(data))
    margin = peak * 0.2 or 0.001
    ax.set_ylim(-peak - margin, peak + margin)

    elapsed = time.time() - plot_start_time
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    ax.set_title(f"Hydrophone — Live Waveform    {h:02d}:{m:02d}:{s:02d}")

    # Draw a dashed line for any new trigger events still in the visible window.
    with trigger_lock:
        new_events         = trigger_events[last_trigger_count:]
        last_trigger_count = len(trigger_events)

    now_perf = time.perf_counter()
    for ev in new_events:
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


def toggle_recording(event):
    global recording, t_recording_start_perf

    if not recording:
        if not timing_ready.is_set():
            print("WARNING: acquisition not ready yet.")
            return
        t_recording_start_perf = time.perf_counter()
        filename = os.path.join(
            WRITE_DIR, datetime.now().strftime("hydrophone_%Y%m%d_%H%M%S.csv")
        )
        write_queue.put(("start", filename, t_ai_start_wall.isoformat()))
        recording = True
        btn_record.label.set_text("■ Stop Recording")
        btn_record.color = "tomato"
        print(f"[ui] recording → {filename}")
    else:
        recording = False
        write_queue.put(("stop",))
        btn_record.label.set_text("● Enable Recording")
        btn_record.color = "limegreen"
        print("[ui] recording stopped.")

    fig.canvas.draw_idle()


# ── Buttons ───────────────────────────────────────────────────────────────────
ax_btn_rec  = plt.axes([0.25, 0.05, 0.25, 0.08])
btn_record  = Button(ax_btn_rec, "● Enable Recording", color="limegreen")
btn_record.on_clicked(toggle_recording)

ax_btn_trig = plt.axes([0.55, 0.05, 0.22, 0.08])
btn_trigger = Button(ax_btn_trig, "⚡ Fire Trigger", color="gold")
btn_trigger.on_clicked(fire_trigger)

# ── Start everything ──────────────────────────────────────────────────────────
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
    daq_thread.join(timeout=3)
    write_queue.put(("quit",))
    csv_thread.join(timeout=30)
    if csv_thread.is_alive():
        print("WARNING: writer thread did not finish cleanly — output may be truncated.")
    print("Done.")