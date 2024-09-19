#!/bin/bash

for i in {1..20}
do
    export TRAIN_ID=$i
    python tools/mod_arch/DNet/experiments/DNet_BoT3_2.py --config-file configs/uncertainty/DNet.yml
done