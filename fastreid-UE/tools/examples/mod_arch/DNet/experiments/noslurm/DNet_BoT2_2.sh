#!/bin/bash

for i in {1..7}
do
    export TRAIN_ID=$i
    python tools/mod_arch/DNet/experiments/DNet_BoT2_2.py --config-file configs/uncertainty/DNet.yml
done