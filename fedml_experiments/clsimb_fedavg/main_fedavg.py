import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import wandb


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from fedml_api.data_preprocessing.ImageNet_raw.data_loader import load_partition_data_ImageNet_raw

from fedml_api.model.basic.resnet_gn import *
from fedml_api.model.basic.resnet import *
from fedml_api.model.basic.resnet_imagenet import *
from fedml_api.model.basic.resnet_s import *
from fedml_api.model.multiexp_model.model import *

from fedml_api.clsimb_fedavg.fedavg_api import FedAvgAPI
from fedml_api.clsimb_fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS

from fedml_api import checkpoint 

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10_lt', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar100',
                        help='data directory')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on clients')
    parser.add_argument('--partition_alpha', type=float, default=0.1, metavar='PA',
                        help='partition alpha (default: 0.1); in paper 0.1 for CIFAR100-LT, 0.05 for ImageNet-LT, 0.5 for CIFAR10-LT')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')
    parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 0.5)')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='learning rate decay (default: 0.1)')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=2, metavar='EP',
                        help='how many epochs will be trained locally')
    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--client_num_per_round', type=int, default=5, metavar='NN',
                        help='number of workers')
    parser.add_argument('--start_comm_round', type=int, default=0,
                        help='where to start')
    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many round of communications we should use; 2000 for CIFARx, 1000 for imageNet')
    parser.add_argument('--expert_join_time', type=float, default=3/5,
                        help='when does the experts join the training process, at comm_round*expert_join_time')
    
    parser.add_argument('--frequency_of_the_test', type=int, default=10,
                        help='the frequency of the algorithms')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='resume from a state, point to the checkpoint file')
    parser.add_argument('--checkpoint_path', type=str, default='/root/Federated-Long-tailed-Learning/checkpoint',
                        help='save a checkpoint to')
    parser.add_argument('--name', type=str, default='Normal_',
                        help='experiments name') 
    parser.add_argument('--imb_factor', type=float, default=0.01, help='ratio of imbalance data; min class/max class; 0.01 for 100, 0.02 for 50 in paper')
    parser.add_argument('--method', type=str, default="train_exp_bsm_esti_global",
    help='fedavg: ce; re-balance strategies: focal, ldam, lade, blsm, ride, ldae; '
         'local re-balance: ldam; global re-balance: ldam_real_global; GPI: ldam_esti_global')

    parser.add_argument('--contrast', action='store_true', help='contrast learning in weak and strong mode')
    parser.add_argument('--debug', action='store_true', help='do not use wandb')
    parser.add_argument('--use_lr', action='store_true', help='filter linear relative network agg')
    parser.add_argument('--bn_wise', action='store_true', help='how to agg bn layer')
    parser.add_argument('--pre_epochs', type=int, default=2, help='num of epochs in pre training')
    parser.add_argument('--reverse_weight', type=float, default=0.01, help='combine mdcs loss with lade loss, the mdcs lossweight')

    # hyperparameter for long-tailed methods
    parser.add_argument('--num_experts', type=int, default=3, help='experts for ride, ldae')
    parser.add_argument('--beta', type=float, default=0.5, help='train expert client select ratio; 0.6 in GBME')

    return parser


def load_data(args, dataset_name, get_bala_dataloader=False):

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if "imagenet32" in dataset_name:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, traindata_cls_counts = load_partition_data_ImageNet(args)

    elif "imagenet224" in dataset_name:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, traindata_cls_counts = load_partition_data_ImageNet_raw(args)
    else:
        if dataset_name == "cifar10" or dataset_name == "cifar10_lt":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100" or dataset_name == "cifar100_lt":
            data_loader = load_partition_data_cifar100
        else:
            data_loader = load_partition_data_cifar10

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, traindata_cls_counts = data_loader(args)

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    # logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None

    # if "ride" in args.method or "bsm" in args.method:
    #     if model_name == "resnet18_gn":
    #         model = ResNet18GNModel(num_classes=output_dim, reduce_dimension=True, use_norm=True, num_experts=args.num_experts, group_norm=8)
    #     elif model_name == "resnet18":
    #         model = ResNet18GNModel(num_classes=output_dim, reduce_dimension=True, use_norm=True, num_experts=args.num_experts, group_norm=0)
    #     elif model_name == "resnet32":
    #         model = ResNet32Model(num_classes=output_dim, reduce_dimension=True, use_norm=True, num_experts=args.num_experts)

    #     # logging.info(model.state_dict().keys())
    #     return model

    if model_name == "resnet18_gn":
        model = resnet18_gn(num_classes=output_dim, group_norm=8)
    elif model_name == "resnet18":
        model = resnet18(num_classes=output_dim)
    elif model_name == "resnet32":
        # model = ResNet32Model(num_classes=output_dim, reduce_dimension=True, use_norm=True, num_experts=args.num_experts)
        model = resnet32(args, num_classes=output_dim)
    elif model_name == "resnet50_gn":
        model = resnet50_gn(num_classes=output_dim)
    elif model_name == "resnet50_in":
        model = resnet50_in(num_classes=output_dim)
    elif model_name == "resnext50":
        model = resnext50_32x4d(num_classes=output_dim)
    else:
        logger.warning("can't find model!")

    return model


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    seed = 0
    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = add_args(argparse.ArgumentParser(description='Federated_Long-tailed_Learning'))
    args = parser.parse_args()

    if args.resume_from: 
        dataset, model, args, device = checkpoint.load_checkpoint(create_model, load_data, logger, args.resume_from)
        # the following need overwrite in checkpoint's args, only in resume mode; immediatly comment these after the start running of resume.
        args.comm_round = 80000
        args.name = "C100_plain_2SingleFIN"
        args.debug=False
        args.contrast=False
        args.count = 10000
        # args.expert_join_time = 3/5

    else:
        logger.info(args)

        # Avoid randomness of cuda, but it will slow down the training
        if "cifar" in args.dataset:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        logger.info(device)

        torch.set_printoptions(threshold=np.inf)
        # load data
        dataset = load_data(args, args.dataset)
        # create model
        model = create_model(args, model_name=args.model, output_dim=dataset[7])

    if not args.debug:
        wandb.init(
            project="Federated_Long-tailed_Learning",
            entity="hongdachen",
            name= str(args.name) + str(args.method) + "-r " + str(args.comm_round) + str(args.dataset) + str(args.partition_method) +
                "-factor" + str(args.imb_factor),
            config=args
        )

    model_trainer = MyModelTrainerCLS(model, args)
    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer)

    fedavgAPI.train()
    if not args.debug:
        wandb.finish()
