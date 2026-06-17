#!/bin/bash
set -e

# Fix ownership of mounted volumes that Docker may create as root
# This runs as root before dropping to dlstreamer
if [ -d "/app/results" ]; then
    chown -R dlstreamer:dlstreamer /app/results
fi

# Drop privileges and exec the CMD as dlstreamer
exec gosu dlstreamer "$@"
