#!/usr/bin/env bash
# Command record, not an unattended deployment script. Run stages serially.
# Baseline launch script and configuration are adjacent; initial preflight:
./run.sh --config /tmp/ds41rt-nvfp4-ab.config --rtx-gpus 1 --dry-run
bash /tmp/ds41rt-nvfp4-baseline-launch.sh
TOKENIZER=/home/tj/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079/tokenizer.json
.venv/bin/python scripts/bench-ds41-release-decode.py --base-url http://127.0.0.1:8000 --tokenizer "$TOKENIZER" --label nvfp4-v7-published-1x-m16-shard1 --repeats 3 --nonce-seed 198473621 --include-counting --output /tmp/nvfp4-v7-baseline-198473621.json
# Stop ALL serving before any export.
docker stop ds41rt-nvfp4-ab
for host in ostrich dodo emu kiwi; do ssh "$host" 'docker stop ds41rt-nvfp4-ab'; done
# Isolate the two opt-in changes; do not change checked-in defaults.
docker exec ds41rt-coordinator-wip bash -lc 'mkdir -p /wip/nvfp4-ab-source; tar -C /wip/source --exclude=.git --exclude=target --exclude=__pycache__ -cf - . | tar -C /wip/nvfp4-ab-source -xf -'
docker cp /tmp/nvfp4-ab-experts.cmake ds41rt-coordinator-wip:/wip/nvfp4-ab-source/native/cmake/v41_nvfp4_experts.cmake
docker exec -e CMAKE_BUILD_PARALLEL_LEVEL=1 ds41rt-coordinator-wip bash -lc '/wip/nvfp4-ab-source/scripts/build-release-artifacts.sh /wip/nvfp4-ab-source coordinator 120 /wip/output/nvfp4-ab-auto-shard5'
# During builds, retain generated v41_nvfp4_ROLE directories after variants.h
# appears. Release script deletes its /tmp/ds41rt-release-build.* tree on exit.
ssh ostrich 'docker exec ds41rt-spark-expert-wip bash -lc "mkdir -p /wip/nvfp4-ab-source; tar -C /wip/source --exclude=.git --exclude=target --exclude=__pycache__ -cf - . | tar -C /wip/nvfp4-ab-source -xf -"'
scp /tmp/nvfp4-ab-experts.cmake ostrich:/tmp/nvfp4-ab-experts.cmake
ssh ostrich 'docker cp /tmp/nvfp4-ab-experts.cmake ds41rt-spark-expert-wip:/wip/nvfp4-ab-source/native/cmake/v41_nvfp4_experts.cmake'
ssh ostrich 'docker exec -e CMAKE_BUILD_PARALLEL_LEVEL=1 ds41rt-spark-expert-wip bash -lc "/wip/nvfp4-ab-source/scripts/build-release-artifacts.sh /wip/nvfp4-ab-source expert 121 /wip/output/nvfp4-ab-auto-shard5"'
# Run numerical + launch tests for each role, rows1/1024/4096, device0,
# sequentially after both builds. Exact template (inside corresponding WIP):
# python3 /wip/nvfp4-ab-source/native/tests/v41_nvfp4_numerics_selftest.py \
#   /wip/output/nvfp4-ab-auto-shard5/libds41rt_native.so "$MANIFEST" \
#   --prefix "$PREFIX" --rows "$ROWS" --device 0 \
#   --sparkinfer /wip/nvfp4-ab-source/third_party/sparkinfer
# python3 /wip/nvfp4-ab-source/native/tests/v41_nvfp4_launch_selftest.py \
#   /wip/output/nvfp4-ab-auto-shard5/libds41rt_native.so "$MANIFEST" \
#   "$PREFIX" --rows "$ROWS" --device 0
# Prefixes: backbone=ds41rt_v41_nvfp4_local_expert,
# tp2=ds41rt_v41_nvfp4_tp2_expert, spark=ds41rt_v41_nvfp4_expert.
docker cp ds41rt-coordinator-wip:/wip/output/nvfp4-ab-auto-shard5/libds41rt_native.so /tmp/nvfp4-ab-coordinator.so
docker cp /tmp/nvfp4-ab-coordinator.so ds41rt-nvfp4-ab:/opt/ds41rt/lib/libds41rt_native.so
ssh ostrich 'docker cp ds41rt-spark-expert-wip:/wip/output/nvfp4-ab-auto-shard5/libds41rt_native.so /tmp/nvfp4-ab-expert.so'
scp ostrich:/tmp/nvfp4-ab-expert.so /tmp/nvfp4-ab-expert.so
for host in ostrich dodo emu kiwi; do
  scp /tmp/nvfp4-ab-expert.so "$host":/tmp/nvfp4-ab-expert.so
  ssh "$host" 'docker cp /tmp/nvfp4-ab-expert.so ds41rt-nvfp4-ab:/opt/ds41rt/lib/libds41rt_native.so'
done
for host in ostrich dodo emu kiwi; do ssh "$host" 'docker start ds41rt-nvfp4-ab'; done
docker start ds41rt-nvfp4-ab
# /v1/models is insufficient readiness: first run below failed all30 streams.
.venv/bin/python scripts/bench-ds41-release-decode.py --base-url http://127.0.0.1:8000 --tokenizer "$TOKENIZER" --label nvfp4-v7-optin-1x-auto-shard5 --repeats 3 --nonce-seed 198473622 --include-counting --output /tmp/nvfp4-v7-optin-198473622.json
# Completed streaming probe before rerun; no artifacts/services changed.
curl -sS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":8,"stream":true,"thinking":{"type":"disabled"}}'
.venv/bin/python scripts/bench-ds41-release-decode.py --base-url http://127.0.0.1:8000 --tokenizer "$TOKENIZER" --label nvfp4-v7-optin-1x-auto-shard5-ready --repeats 3 --nonce-seed 198473623 --include-counting --output /tmp/nvfp4-v7-optin-198473623.json
