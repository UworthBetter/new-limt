#!/usr/bin/env bash
# Watch both split-seed kernels; fetch each output as it completes.
declare -A DONE
while [ ${#DONE[@]} -lt 2 ]; do
  for k in csi500-h-s1 csi500-h-s23; do
    [ -n "${DONE[$k]}" ] && continue
    s=$(kaggle kernels status beelingz/$k 2>/dev/null)
    echo "$(date '+%F %T') $k: $s"
    case "$s" in
      *COMPLETE*|*ERROR*|*CANCEL*)
        mkdir -p /d/research/new_LiMT/runs/kaggle_incoming/$k
        kaggle kernels output beelingz/$k -p /d/research/new_LiMT/runs/kaggle_incoming/$k 2>&1
        DONE[$k]=1 ;;
    esac
  done
  sleep 600
done
echo SPLIT_POLL_DONE
