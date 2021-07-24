#!/bin/bash

expt_dir="/cds/data/psdm/amo/amo86615/"
echo "Checking $expt_dir"
now=$(date +"%T")
echo "Current time : $now"
#
rstart=182; rend=198; run=$rstart
while [ $run -lt $rend ]
do 
  nxtc=`ls ${expt_dir}xtc/*r0${run}*.xtc 2>/dev/null | wc -l `
  nidx=`ls ${expt_dir}xtc/index/*r0${run}*.idx 2>/dev/null | wc -l `
  echo "Run: $run >   # of XTC: $nxtc >   # of IDX: $nidx"
  run=`expr $run + 1`
done
