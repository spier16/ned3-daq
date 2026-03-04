import glob
import json
import os

import numpy as np
import pandas as pd


SEARCH_DIRS = [
    os.path.expanduser("~/Desktop"),
    os.getcwd(),
]


def find_latest_matched_pair():
    for base in SEARCH_DIRS:
        sync_files = sorted(
            glob.glob(os.path.join(base, "ttl_validation_*_sync.json")),
            key=os.path.getmtime,
            reverse=True,
        )
        for sync_path in sync_files:
            csv_path = sync_path.replace("_sync.json", ".csv")
            if os.path.exists(csv_path):
                return csv_path, sync_path
    raise FileNotFoundError(
        "Could not find a matched ttl_validation_*.csv + ttl_validation_*_sync.json pair.\n"
        f"Searched: {SEARCH_DIRS}"
    )


csv_path, sync_path = find_latest_matched_pair()
print(f"using csv : {csv_path}")
print(f"using sync: {sync_path}")

df = pd.read_csv(csv_path, comment="#")
with open(sync_path, "r") as f:
    sync = json.load(f)

frac_sample = sync["trigger"]["fractional_sample_index"]
csv_time0 = sync["recording"]["csv_time0_hardware_sample_index"]
sample_rate = sync["recording"]["sample_rate_hz"]

print("--- sync JSON diagnostics ---")
print(
    f"fractional_sample_index        : {frac_sample:.3f}  "
    f"({frac_sample / sample_rate * 1e3:.3f} ms after AI start)"
)
print(
    f"csv_time0_hardware_sample_index: {csv_time0}  "
    f"({csv_time0 / sample_rate * 1e3:.3f} ms of pre-recording acquisition)"
)
print(
    "time_since_recording_start_s   : "
    f"{sync['trigger'].get('time_since_recording_start_s', 'N/A')}"
)
print(f"rollover_count                 : {sync['trigger'].get('rollover_count', 'N/A')}")
print(f"true_tick_count                : {sync['trigger'].get('true_tick_count', 'N/A')}")
print(
    "hardware_samples_read_at_trigger: "
    f"{sync['trigger'].get('hardware_samples_read_at_trigger', 'N/A')}"
)
print("-----------------------------")

predicted_row = frac_sample - csv_time0

# Find where voltage crosses 1.5 V on the rising edge.
measured_row = int(np.argmax(df["voltage_V"].values > 1.5))

residual_us = (predicted_row - measured_row) / sample_rate * 1e6
residual_samples = predicted_row - measured_row

print("--- row comparison (JSON vs CSV) ---")
print(
    "predicted row (JSON): "
    f"{predicted_row:.3f}  = fractional_sample_index - csv_time0_hardware_sample_index"
)
print(
    "measured row (CSV) : "
    f"{measured_row}  = first sample where voltage_V > 1.5 V"
)
print(
    "residual (JSON - CSV): "
    f"{residual_samples:+.3f} samples  ({residual_us:+.1f} us)"
)
if residual_us < 0:
    print("interpretation: negative residual => JSON prediction is earlier than CSV edge.")
elif residual_us > 0:
    print("interpretation: positive residual => JSON prediction is later than CSV edge.")
else:
    print("interpretation: zero residual => JSON prediction and CSV edge coincide.")
