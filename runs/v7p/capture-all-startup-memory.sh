#!/usr/bin/env bash
# Serve each configuration that is not yet recorded, capture its startup latency
# and post-readiness memory, then stop it. exl3-2x was already recorded during
# the campaign while it was serving for the evaluation.
set -uo pipefail

cd /home/tj/Developer/ds41rt
SERVE=runs/v7p/serve-published.sh
for config in nvfp4-1x nvfp4-2x exl3-5090; do
  echo "=== $config ==="
  if ! bash "$SERVE" "$config" start > "/tmp/capture-$config-serve.log" 2>&1; then
    echo "$config FAILED TO SERVE"; tail -5 "/tmp/capture-$config-serve.log"; continue
  fi
  bash runs/v7p/capture-startup-memory.sh "$config" || echo "$config capture failed"
  bash "$SERVE" "$config" stop >/dev/null 2>&1
  sleep 5
done
echo "STARTUP/MEMORY CAPTURE DONE"
