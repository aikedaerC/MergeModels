# Federated-Long-tailed-Learning

This repository is the official implementation of ["Global Balanced Experts for Federated Long-Tailed Learning"](https://openaccess.thecvf.com/content/ICCV2023/papers/Zeng_Global_Balanced_Experts_for_Federated_Long-Tailed_Learning_ICCV_2023_paper.pdf)

## Installation

```
conda env create -f ldae.yaml
```

## Run

```
cd fedml_experiments/clsimb_fedavg/
```
For bsm on cifar100-lt, run:
```
python -u ./main_fedavg.py \
--comm_round 2500 --epochs 2 --batch_size 64 --client_optimizer sgd --lr 0.6 --lr_decay 0.05 \
--imb_factor 0.01 --partition_alpha 0.1 --method train_exp_bsm_esti_global --expert_join_time 0 \
--num_experts 1 --frequency_of_the_test 50 --beta 0.8 --model resnet32 --client_num_in_total 100 \
--client_num_per_round 20 
```

Actually:

C100
```
python -u ./main_fedavg.py --comm_round 2100 --epochs 2 --batch_size 64 \
--client_optimizer sgd --lr 0.6 --lr_decay 0.05 --imb_factor 0.02 --dataset cifar100_lt \
--partition_alpha 0.1 --method lade_real_global --frequency_of_the_test 50 --beta 0.8 \
--model resnet32 --client_num_in_total 100 --client_num_per_round 20 --name C100_Last --use_lr --debug
```

C10
```
python -u ./main_fedavg.py --comm_round 2500 --epochs 2 --batch_size 64 \
--client_optimizer sgd --lr 0.1 --lr_decay 0.05 --imb_factor 0.02 --dataset cifar10_lt \
--partition_alpha 0.5 --method lade_real_global --frequency_of_the_test 50 --beta 0.8 \
--model resnet18 --client_num_in_total 10 --client_num_per_round 5 --name C10
```

PretrainingWith
```
python -u ./main_fedavg.py --comm_round 1250 --epochs 2 --batch_size 64 \
--client_optimizer sgd --lr 0.6 --lr_decay 0.05 --imb_factor 0.02 --dataset cifar100_lt \
--partition_alpha 0.1 --method lade_esti_global --frequency_of_the_test 50 --beta 0.8 \
--model resnet32 --client_num_in_total 100 --client_num_per_round 20 --name C100CON_Less \
--contrast --pre_epochs 10 --reverse_weight 0
```

Other re-balance strategies are available: focal, ldam, lade, blsm, ride.
They can use different class priors like: local re-balance: ldam; global re-balance: ldam_real_global; GPI: ldam_esti_global.

```
python -u ./main_fedavg.py --comm_round 2500 --epochs 2 --batch_size 16 \
--client_optimizer sgd --lr 0.6 --lr_decay 0.05 --imb_factor 0.02 --dataset cifar100_lt \
--partition_alpha 0.1 --method lade_esti_global --frequency_of_the_test 2 --beta 0.8 \
--model resnet32 --client_num_in_total 100 --client_num_per_round 20 --name C100CON \
--contrast --pre_epochs 1 --debug 
```

resume
```
python -u ./main_fedavg.py --resume_from /root/Federated-Long-tailed-Learning/checkpoint/C100_plain/round_1600_global.pth
```