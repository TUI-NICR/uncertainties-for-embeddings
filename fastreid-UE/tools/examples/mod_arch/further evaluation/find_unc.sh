#!/bin/bash


/home/ange8547/anaconda3/envs/UE/bin/python /home/ange8547/UE/fastreid-UE/tools/mod_arch/further\ evaluation/find_uncertain_images.py --config-file /home/ange8547/UE/fastreid-UE/configs/uncertainty/DNet.yml MODEL.WEIGHTS /results_nas/ange8547/mod_arch/DNet/bb_naive/naive/0/model_final.pth

/home/ange8547/anaconda3/envs/UE/bin/python /home/ange8547/UE/fastreid-UE/tools/mod_arch/further\ evaluation/find_uncertain_images.py --config-file /home/ange8547/UE/fastreid-UE/configs/uncertainty/PFE.yml MODEL.WEIGHTS /results_nas/ange8547/mod_arch/PFE/first_try/3/model_final.pth

/home/ange8547/anaconda3/envs/UE/bin/python /home/ange8547/UE/fastreid-UE/tools/mod_arch/further\ evaluation/find_uncertain_images.py --config-file /home/ange8547/UE/fastreid-UE/configs/uncertainty/UAL.yml MODEL.WEIGHTS /results_nas/ange8547/mod_arch/UAL/Dropout/6/model_final.pth
