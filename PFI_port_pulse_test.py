import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from collections import deque
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
DEVICE           = "cDAQ3"
COUNTER_OUT_CHAN = f"{DEVICE}/ctr0"
COUNTER_IN_CHAN  = f"{DEVICE}/ctr1"
PFI0_TERMINAL    = f"/{DEVICE}/PFI0"   # output — wire this to PFI1
PFI1_TERMINAL    = f"/{DEVICE}/PFI1"   # input  — wire this from PFI0
PULSE_HIGH_TIME  = 0.05                # 50 ms pulse width
POLL_INTERVAL    = 0.02                # 20 ms polling loop on PFI1
MAX_LOG_LINES    = 20

# ── Shared state ───────────────────────────────────────────────────────────────
stop_event     = threading.Event()
last_count     = 0
total_sent     = 0
total_detected = 0
count_lock     = threading.Lock()
log_lock       = threading.Lock()
log_lines      = deque(maxlen=MAX_LOG_LINES)
log_dirty      = threading.Event()

indicator_color = ["gray"]   # "gray" | "green" | "red"  — set by threads


def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(msg: str, level: str = "info"):
    prefix = {"info": "   ", "ok": "✓  ", "warn": "⚠  ", "err": "✗  "}.get(level, "   ")
    line = f"[{_ts()}] {prefix}{msg}"
    with log_lock:
        log_lines.append(line)
    print(line)
    log_dirty.set()


# ── Pulse output ───────────────────────────────────────────────────────────────
def fire_pulse(_event=None):
    """Button callback — spawns a daemon thread so the UI stays responsive."""
    threading.Thread(target=_do_fire, daemon=True).start()


def _do_fire():
    global total_sent
    with count_lock:
        total_sent += 1
        n = total_sent

    log(f"Firing pulse #{n} on PFI0  ({PULSE_HIGH_TIME*1000:.0f} ms high)…")
    try:
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
                PFI0_TERMINAL,
            )
            co_task.start()
            co_task.wait_until_done(timeout=max(PULSE_HIGH_TIME * 4, 2.0))

        log(f"Pulse #{n} output complete.", "ok")
    except Exception as exc:
        log(f"Error firing pulse: {exc}", "err")


# ── PFI1 monitoring thread ─────────────────────────────────────────────────────
def monitor_thread():
    global last_count, total_detected

    log("Starting PFI1 monitor (ctr1)…")
    try:
        with nidaqmx.Task() as ci_task:
            ci_chan = ci_task.ci_channels.add_ci_count_edges_chan(
                COUNTER_IN_CHAN,
                edge=Edge.RISING,
                initial_count=0,
            )
            ci_chan.ci_count_edges_term = PFI1_TERMINAL   # ← direct reference

            ci_task.start()
            log(f"Monitoring {PFI1_TERMINAL} for rising edges.  "
                f"Wire PFI0 → PFI1, then press 'Fire Pulse'.", "ok")

            while not stop_event.is_set():
                try:
                    hw_count = ci_task.read()
                except nidaqmx.errors.DaqError as e:
                    log(f"Read error: {e}", "err")
                    break

                with count_lock:
                    delta = hw_count - last_count
                    if delta > 0:
                        total_detected += delta
                        last_count = hw_count
                        d, s = total_detected, total_sent

                if delta > 0:
                    indicator_color[0] = "green"
                    log(
                        f"PULSE DETECTED on PFI1  "
                        f"(edge #{d}  |  {d}/{s} pulses caught)",
                        "ok",
                    )
                    threading.Timer(0.4, lambda: indicator_color.__setitem__(0, "gray")).start()

                time.sleep(POLL_INTERVAL)

    except Exception as exc:
        log(f"Monitor thread error: {exc}", "err")
        indicator_color[0] = "red"

    log("Monitor thread exited.")


# ── Matplotlib UI ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 5))
fig.suptitle("PFI0 → PFI1 Loopback Test  |  NI cDAQ-9178", fontsize=12, fontweight="bold")
plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.18)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4)

# ── Indicator circle ──
ax_ind = fig.add_subplot(gs[0, 0])
ax_ind.set_aspect("equal")
ax_ind.axis("off")
indicator_patch = plt.Circle((0.5, 0.5), 0.35, color="gray", transform=ax_ind.transAxes)
ax_ind.add_patch(indicator_patch)
ind_label = ax_ind.text(0.5, 0.08, "Waiting…", ha="center", va="bottom",
                        transform=ax_ind.transAxes, fontsize=9)

# ── Counters ──
ax_cnt = fig.add_subplot(gs[0, 1])
ax_cnt.axis("off")
counter_text = ax_cnt.text(
    0.05, 0.85,
    "Pulses sent    : 0\nPulses detected: 0\nSuccess rate   : —",
    transform=ax_cnt.transAxes,
    fontsize=11, fontfamily="monospace",
    verticalalignment="top",
)

# ── Log area ──
ax_log = fig.add_subplot(gs[1, :])
ax_log.axis("off")
log_text_obj = ax_log.text(
    0.01, 0.99, "",
    transform=ax_log.transAxes,
    fontsize=7.5, fontfamily="monospace",
    verticalalignment="top",
    wrap=False,
)

# ── Buttons ──
ax_fire  = plt.axes([0.30, 0.04, 0.20, 0.09])
btn_fire = Button(ax_fire, "⚡  Fire Pulse", color="gold")
btn_fire.on_clicked(fire_pulse)

ax_quit  = plt.axes([0.55, 0.04, 0.15, 0.09])
btn_quit = Button(ax_quit, "✕  Quit", color="salmon")
btn_quit.on_clicked(lambda _: plt.close("all"))


def _update_ui(_frame=None):
    # Indicator
    with count_lock:
        sent = total_sent
        det  = total_detected

    color = indicator_color[0]
    indicator_patch.set_color(color)
    labels = {"gray": "Waiting…", "green": "DETECTED ✓", "red": "Error ✗"}
    ind_label.set_text(labels.get(color, ""))

    # Counters
    rate = f"{det/sent*100:.0f} %" if sent else "—"
    counter_text.set_text(
        f"Pulses sent    : {sent}\n"
        f"Pulses detected: {det}\n"
        f"Success rate   : {rate}"
    )

    # Log
    if log_dirty.is_set():
        with log_lock:
            text = "\n".join(log_lines)
        log_text_obj.set_text(text)
        log_dirty.clear()

    fig.canvas.draw_idle()


import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, _update_ui, interval=100,
                              blit=False, cache_frame_data=False)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mon = threading.Thread(target=monitor_thread, daemon=True)
    mon.start()

    try:
        plt.show()
    finally:
        stop_event.set()
        mon.join(timeout=3)
        print("Done.")