#!/bin/bash
set -e

# Fix ownership of mounted volumes that Docker may create as root
# This runs as root before dropping to appuser
if [ -d "/app/results" ]; then
    chown -R appuser:appuser /app/results
fi

# Drop privileges and exec the CMD as appuser
exec gosu appuser "$@"
