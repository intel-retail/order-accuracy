'''
* Copyright (C) 2025 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import os
import time
from datetime import datetime

NPU_PATH = os.getenv("NPU_PATH", "/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us")
NPU_LOG = os.getenv("NPU_LOG", "npu_usage.csv")

def read_npu_runtime(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except:
        return None

def main():
    print(f"Logging NPU usage to '{NPU_LOG}' (Ctrl+C to stop)...")

    prev_runtime = read_npu_runtime(NPU_PATH)
    prev_time = time.monotonic()
    if prev_runtime is None:
        print("Initial NPU read failed. Exiting.")
        return

    with open(NPU_LOG, "w", buffering=1) as log:
        log.write("timestamp,percent_usage\n")
        try:
            while True:
                time.sleep(1)
                curr_runtime = read_npu_runtime(NPU_PATH)
                curr_time = time.monotonic()
                if curr_runtime is None:
                    continue
                usage = (curr_runtime - prev_runtime) / ((curr_time - prev_time) * 1e6) * 100
                log.write(f"{datetime.now().isoformat()},{usage:.2f}\n")
                prev_runtime, prev_time = curr_runtime, curr_time
        except KeyboardInterrupt:
            print("\nStopped logging.")

if __name__ == "__main__":
    main()
