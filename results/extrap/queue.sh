#!/bin/bash
# waits for phase B, then runs complex split then low-data curve, AMN-NS and MLP, 3 seeds
cd /Users/jean-loup-faulon/Desktop/Hybdid/AMN-codes/amn_ns
while pgrep -f "amn_[n]ullspace.py --model .* --tag B_" >/dev/null; do sleep 120; done
P=/usr/local/bin/python3
for s in 1 2 3; do
  G=0; [ $s = 1 ] && G=1
  AMN_USE_GPU=$G TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 bash -c "
    for sp in complex lowdata; do
      $P -u amn_nullspace.py --model ns  --split \$sp --param null --plateau --epochs 100 --seed $s --tag \${sp}_ns_s$s  --out results/extrap
      $P -u amn_nullspace.py --model mlp --split \$sp --seed $s --tag \${sp}_mlp_s$s --out results/extrap
    done" > results/extrap/s$s.log 2>&1 &
done
wait
date > results/extrap/finished
