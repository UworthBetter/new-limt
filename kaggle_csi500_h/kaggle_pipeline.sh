#!/usr/bin/env bash
# Autonomous Kaggle pipeline: wait for dataset v2 files -> push GPU kernel ->
# poll status -> fetch output -> verify manifest hashes locally.
cd /d/research/new_LiMT/kaggle_csi500_h
until kaggle datasets files beelingz/csi500-newlimt 2>/dev/null | grep -q new_v2_panel; do sleep 180; done
sleep 300
cd kernel && kaggle kernels push 2>&1; cd ..
while true; do
  s=$(kaggle kernels status beelingz/csi500-h-kaggle 2>/dev/null)
  echo "$(date '+%F %T') kernel: $s"
  if echo "$s" | grep -qiE '"complete"|^complete|complete$'; then break; fi
  if echo "$s" | grep -qiE 'error|cancel'; then echo "KERNEL FAILED"; break; fi
  sleep 600
done
mkdir -p /d/research/new_LiMT/runs/kaggle_incoming/csi500_H
kaggle kernels output beelingz/csi500-h-kaggle -p /d/research/new_LiMT/runs/kaggle_incoming/csi500_H 2>&1
echo KAGGLE_PIPELINE_DONE
