#!/usr/bin/env bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

# initial setup
(
    cd ../
    make clean
    sleep 3
    make build-benchmark-docker
)

rm -rf results || true

# Run benchmark on test docker compose file
python3 benchmark.py --retail_use_case_root test_src/ --compose_file test_src/docker-compose.yml
# consolidate results
make consolidate
