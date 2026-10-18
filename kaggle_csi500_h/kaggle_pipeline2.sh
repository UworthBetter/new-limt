#!/usr/bin/env bash
# Fixed poller for kernel v2: case-based completion detection + output fetch.
while true; do
  s=$(kaggle kernels status beelingz/csi500-h-kaggle 2>/dev/null)
  echo "$(date '+%F %T') kernel: $s"
  case "$s" in
    *COMPLETE*|*ERROR*|*CANCEL*) break ;;
  esac
  sleep 600
done
mkdir -p /d/research/new_LiMT/runs/kaggle_incoming/csi500_H_v2
kaggle kernels output beelingz/csi500-h-kaggle -p /d/research/new_LiMT/runs/kaggle_incoming/csi500_H_v2 2>&1
echo KAGGLE_PIPELINE_V2_DONE
