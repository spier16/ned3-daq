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

# ── Configuration ────────────────────────────────────────────────────────────
CHANNEL       = "cDAQ3Mod7/ai2"   # update to match your device name
SAMPLE_RATE   = 12800                 # Hz
BUFFER_SIZE   = 1280                  # samples per read (100ms chunks)
VOLTAGE_RANGE = 5                 # voltage range (±5 V in AC coupling mode)
IEPE_ENABLED  = False                 # set True for hydrophones requiring IEPE excitation
PLOT_WINDOW   = 1.0                   # seconds of data shown in live plot
WRITE_DIR     = r"C:\Users\sapierso\Desktop"

# ── Shared state ─────────────────────────────────────────────────────────────
recording    = False
write_queue  = queue.Queue(maxsize=200)  # ~20 s of headroom at 12800 Hz / 1280-sample chunks
plot_buffer  = np.zeros(int(SAMPLE_RATE * PLOT_WINDOW))
buffer_lock  = threading.Lock()
stop_event   = threading.Event()


# ── Writer thread (consumes queue, does all disk I/O) ─────────────────────────
def writer_thread():
    csv_file     = None
    sample_count = 0

    while True:
        item = write_queue.get()

        if item[0] == "start":
            try:
                csv_file     = open(item[1], "w")
                csv_file.write("time_s,voltage_V\n")
                sample_count = 0
            except OSError as e:
                print(f"ERROR: Could not open recording file: {e}")
                csv_file = None
        elif item[0] == "data":
            if csv_file:
                try:
                    samples = item[1]
                    t0      = sample_count / SAMPLE_RATE
                    times   = t0 + np.arange(len(samples)) / SAMPLE_RATE
                    sample_count += len(samples)
                    csv_file.write(
                        "\n".join(f"{t:.6f},{v:.6f}" for t, v in zip(times, samples)) + "\n"
                    )
                except OSError as e:
                    print(f"ERROR: Failed to write samples to CSV: {e}")
        elif item[0] == "stop":
            if csv_file:
                csv_file.close()
                csv_file = None
        elif item[0] == "quit":
            if csv_file:
                csv_file.close()
            break

        write_queue.task_done()


# ── DAQ acquisition thread ───────────────────────────────────────────────────
def acquisition_thread():
    global plot_buffer

    with nidaqmx.Task() as task:
        ch = task.ai_channels.add_ai_voltage_chan(
            CHANNEL,
            terminal_config=TerminalConfiguration.PSEUDO_DIFF,
            min_val=-VOLTAGE_RANGE,
            max_val=VOLTAGE_RANGE,
        )

        ch.ai_coupling = Coupling.AC

        # IEPE (constant current) excitation for active hydrophones
        if IEPE_ENABLED:
            ch.ai_excit_src = ExcitationSource.INTERNAL
            ch.ai_excit_val = 0.004  # 4 mA

        task.timing.cfg_samp_clk_timing(
            rate=SAMPLE_RATE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=BUFFER_SIZE * 4,
        )

        task.start()
        print("Acquisition started.")

        while not stop_event.is_set():
            samples = np.array(
                task.read(number_of_samples_per_channel=BUFFER_SIZE, timeout=2.0)
            )

            # Update rolling plot buffer
            with buffer_lock:
                plot_buffer = np.roll(plot_buffer, -len(samples))
                plot_buffer[-len(samples):] = samples

            # Hand samples off to the writer thread — no disk I/O here
            if recording:
                try:
                    write_queue.put_nowait(("data", samples))
                except queue.Full:
                    print("WARNING: Write queue full — samples dropped. Disk may be too slow.")

        print("Acquisition stopped.")


# ── UI / plotting ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(bottom=0.2)

time_axis = np.linspace(-PLOT_WINDOW, 0, int(SAMPLE_RATE * PLOT_WINDOW))
(line,)   = ax.plot(time_axis, plot_buffer, lw=0.8, color="dodgerblue")

ax.set_xlim(-PLOT_WINDOW, 0)
ax.set_ylim(-VOLTAGE_RANGE, VOLTAGE_RANGE)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (V)")
ax.set_title("Hydrophone — Live Waveform")
ax.grid(True, alpha=0.3)

plot_start_time = time.time()


def animate(_frame):
    with buffer_lock:
        data = plot_buffer.copy()

    line.set_ydata(data)

    peak = np.max(np.abs(data))
    margin = peak * 0.2 or 0.001  # 20% padding, fallback if signal is silent
    ax.set_ylim(-peak - margin, peak + margin)

    elapsed = time.time() - plot_start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    ax.set_title(f"Hydrophone — Live Waveform    {h:02d}:{m:02d}:{s:02d}")

    return (line,)


def toggle_recording(event):
    global recording

    if not recording:
        filename = os.path.join(
            WRITE_DIR, datetime.now().strftime("hydrophone_%Y%m%d_%H%M%S.csv")
        )
        write_queue.put(("start", filename))  # file must be open before data arrives
        recording = True                       # only now start queuing data
        btn_record.label.set_text("■ Stop Recording")
        btn_record.color = "tomato"
        print(f"Recording → {filename}")
    else:
        recording = False          # stop queuing data before signalling the writer
        write_queue.put(("stop",))
        btn_record.label.set_text("● Enable Recording")
        btn_record.color = "limegreen"
        print("Recording stopped.")

    fig.canvas.draw_idle()


ax_btn = plt.axes([0.35, 0.05, 0.3, 0.08])
btn_record = Button(ax_btn, "● Enable Recording", color="limegreen")
btn_record.on_clicked(toggle_recording)

# ── Start everything ──────────────────────────────────────────────────────────
daq_thread = threading.Thread(target=acquisition_thread, daemon=True)
daq_thread.start()

csv_thread = threading.Thread(target=writer_thread, daemon=True)
csv_thread.start()

ani = animation.FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)

try:
    plt.show()
finally:
    recording = False          # stop queuing data
    stop_event.set()
    daq_thread.join(timeout=3) # wait for acquisition to stop before draining queue
    write_queue.put(("quit",)) # writer processes all remaining data, then exits
    csv_thread.join(timeout=30)
    if csv_thread.is_alive():
        print("WARNING: Writer thread did not finish cleanly — output file may be truncated.")
    print("Done.")