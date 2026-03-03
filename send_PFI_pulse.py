import nidaqmx
from nidaqmx.constants import AcquisitionType, Signal

DEVICE = "cDAQ3"
PFI_PORT_NO = "0"
PULSE_HIGH_TIME = 2.0

with nidaqmx.Task() as co_task:
    co_task.co_channels.add_co_pulse_chan_time(
        f"{DEVICE}/ctr0",
        low_time=1e-6,
        high_time=PULSE_HIGH_TIME,
    )
    co_task.timing.cfg_implicit_timing(
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=1,
    )
    co_task.export_signals.export_signal(
        Signal.COUNTER_OUTPUT_EVENT,
        f"/{DEVICE}/PFI{PFI_PORT_NO}",
    )
    print(f"Firing {PULSE_HIGH_TIME}-second pulse on PFI{PFI_PORT_NO}...")
    co_task.start()
    co_task.wait_until_done(timeout=5.0)
    print("Done.")