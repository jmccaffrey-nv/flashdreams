#!/usr/bin/env bash
set -euo pipefail
validation=/home/jmccaffrey/projects/artifacts/flashdreams-pr773-validation
source_root=/home/jmccaffrey/projects/flashdreams-restore-drive-controls
python=/home/jmccaffrey/projects/cosmos3-sim-drag-race-2026-10-23/src/flashdreams-v0.2.0/.venv/bin/python
scene=/home/jmccaffrey/.cache/flashdreams/cosmos3-nano-sim-auto/scenes/clipgt-residential-iter1500-e78981609da88c7e6d2c68bb30caacd71fdee21c27e35431f6a77538669568ed
checkpoint=/home/jmccaffrey/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano-Sim-Auto/snapshots/c3fa1378e93710242501a6288d7c9e7c9084a580/model.safetensors.index.json
mode=${1:?open-loop or closed-loop}
cd "$source_root"
if [[ "$mode" == closed-loop ]]; then
  application=cosmos3_nano_sim_auto.apps.interactive_drive.adapter
  steps=13
  presentation=continuous
  application_args=(--scene "$scene" --checkpoint "$checkpoint" --views 1 --compile-denoise off --total-blocks 13)
elif [[ "$mode" == open-loop ]]; then
  application=cosmos3_nano_sim_auto.impl.app
  steps=5
  presentation=on_demand
  application_args=(--sample "$scene" --draw-map "$scene" --checkpoint "$checkpoint" --views 1 --frames 49 --compile-denoise off --ui)
else
  exit 2
fi
exec "$python" -u "$validation/capture.py" "$mode" "$application" \
  --mode mp4 --output-path "$validation/$mode/drive.mp4" \
  --stats-path "$validation/$mode/metrics.json" --timeout unbound \
  --total-model-steps "$steps" --pixel-width 1280 --pixel-height 800 \
  --backpressure-mode block --presentation-mode "$presentation" \
  -- "${application_args[@]}"
