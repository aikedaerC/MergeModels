import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import wandb
from tqdm import tqdm
import torch.optim as optim
from .model_trainer import ModelTrainer
from .lossfns import *
# from .multi_experts.trainer import Trainer as MultiExTrainer
# from fedml_api.model.multiexp_model.ldam_drw_resnets.expert_resnet_cifar import NormedLinear


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.extractor_model = None
        self.class_num = None
        self.class_dist = None
        self.islt = False
        self.mixtrain_flag = True
        self.total_cls_num = None
        self.class_range = None
        self.training_exp = None
        self.criterions = {}
        self.criterion_weights = {}


    # for long-tail dataset
    def set_ltinfo(self, class_num=None, mixtrain_flag=None, class_dist=None, class_range=None):
        if class_num is not None:
            self.class_num = class_num

        if self.args is None or "lt" in self.args.dataset:
            self.islt = True
            if mixtrain_flag is not None:
                self.mixtrain_flag = mixtrain_flag
            if class_dist is not None:
                self.class_dist = class_dist
            if class_range is not None:
                self.class_range = class_range

    def wandb_watch_model(self):
        wandb.watch(self.model)

    def get_model_params(self):
        if self.classifer is None:
            return self.model.cpu().state_dict(), None
        return self.model.cpu().state_dict(), self.classifer.cpu().state_dict()

    def set_model_params(self, model_parameters, classifer_param):
        if self.classifer is None:
            return self.model.load_state_dict(model_parameters), None
        return self.model.load_state_dict(model_parameters), self.classifer.load_state_dict(classifer_param)
    
    def get_model(self):
        return self.model, self.classifer

    def set_acc_in_weight(self, cls_acc_metrics, label_smaple_num, device):

        for label in range(self.class_num):
            if label_smaple_num[label] != 0:
                cls_acc_metrics[label] = cls_acc_metrics[label] / label_smaple_num[label]

        # logging.info("cls_acc_metrics in client" + str(cls_acc_metrics))
        model_para,_ = self.get_model_params()
        fc_weight = model_para['fc.weight']

        for i in range(self.class_num):
            fc_weight[i][0] = cls_acc_metrics[i] * 0.1

        self.set_model_params(model_para, None)
        self.model.to(device)

    def train(self, train_data, device, args, alpha=None, cls_num_list=None, round=0):
        model = self.model
        model.to(device)
        model.train()
        # import pdb;pdb.set_trace()
        criterion, optimizer = self.train_init(device, cls_num_list, alpha)


        epoch_loss = []

        train_data.dataset.target = train_data.dataset.target.astype(np.int64)

        cls_acc_metrics = dict.fromkeys(range(self.class_num), 0)
        label_smaple_num = dict.fromkeys(range(self.class_num), 0)

        self.train_count = 0

        for epoch in range(args.epochs):

            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                model.zero_grad()
                extra_info = {}
                if isinstance(x, list):
                    if "contrast" in self.args.method:
                        x[0], x[1], labels = x[0].to(device), x[1].to(device), labels.to(device)
                        data = torch.cat([x[0], x[1]], dim=0).cuda() 
                        target = torch.cat([labels, labels], dim=0).cuda()
                        output,logits = model(data)
                        extra_info.update({
                            "logits": logits.transpose(0, 1),
                            "epoch":epoch
                        })
                        if "lade" in args.method:
                            perform_loss = criterion["perform"](output, labels)
                            routeweight_loss = criterion["routeweight"](output, labels)
                            reverse_loss = criterion["reverse"](output_logits=output, target=labels, extra_info=extra_info)
                            loss = perform_loss + args.lade_weight * routeweight_loss + args.reverse_weight * reverse_loss
                        elif "ldam" in args.method:
                            loss = criterion(output, labels, round, device)
                        else:
                            loss = criterion(output_logits=output, 
                                            target=target, 
                                            extra_info=extra_info)
                    else:
                        x[0], x[1], labels = x[0].to(device), x[1].to(device), labels.to(device)
                        data = torch.cat([x[0], x[1]], dim=0).cuda() 
                        target = torch.cat([labels, labels], dim=0).cuda()
                        log_probs, logits = model(data)

                        log_probs1 = torch.chunk(log_probs, chunks=2, dim=0)[0].squeeze(0)
                        extra_info.update({
                            "logits": logits.transpose(0, 1),
                            "epoch":epoch
                        })

                        if "lade" in args.method:
                            perform_loss = criterion["perform"](log_probs1, labels)
                            routeweight_loss = criterion["routeweight"](log_probs1, labels)
                            reverse_loss = criterion["reverse"](output_logits=log_probs, target=target, extra_info=extra_info)
                            loss = perform_loss + args.lade_weight * routeweight_loss + args.reverse_weight * reverse_loss
                        elif "ldam" in args.method:
                            loss = criterion(log_probs1, labels, round, device)
                        else:
                            loss = criterion(log_probs1, labels)
                else:
                    x, labels = x.to(device), labels.to(device)
                    log_probs, _ = model(x)

                    if "lade" in args.method:
                        perform_loss = criterion["perform"](log_probs, labels)
                        routeweight_loss = criterion["routeweight"](log_probs, labels)
                        # reverse_loss = criterion["reverse"](output_logits=log_probs, target=target, extra_info=extra_info)
                        loss = perform_loss + args.lade_weight * routeweight_loss #+ args.reverse_weight * reverse_loss
                    elif "ldam" in args.method:
                        loss = criterion(log_probs, labels, round, device)
                    else:
                        loss = criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

                self.train_count += 1

                if not isinstance(x, list): 
                    if self.islt and epoch == args.epochs - 1:
                        _, predicted = torch.max(log_probs, -1)
                        correct = predicted.eq(labels)

                        for (idx, label) in enumerate(labels):
                            if correct[idx]:
                                cls_acc_metrics[int(label.item())] += 1
                            label_smaple_num[int(label.item())] += 1

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=20,
                                              gamma=0.1)
        return optimizer, scheduler

    def batch_forward (self, inputs, labels=None, centroids=False, feature_ext=False, phase='train'): 
        '''
        This is a general single batch running function. 
        '''

        # Calculate Features
        _, self.features = self.model(inputs)

        # If not just extracting features, calculate logits
        if not feature_ext:

            # During training, calculate centroids if needed to 
            if phase != 'test':
                if centroids and 'FeatureLoss' in self.criterions.keys():
                    self.centroids = self.criterions['FeatureLoss'].centroids.data
                else:
                    self.centroids = None

            # Calculate logits with classifier
            self.logits, self.direct_memory_feature = self.classifer(self.features, self.centroids)

    def batch_backward(self): 
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers
        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):

        # First, apply performance loss
        self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels) \
                    * self.criterion_weights['PerformanceLoss']

        # Add performance loss to total loss
        self.loss = self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat

    def fintune(self, train_data, device, args, alpha=None, cls_num_list=None, round=0):
        self.model.to(device)
        self.classifer.to(device)
        self.model.train()
        self.classifer.train()

        model_optim_params_list = []
        model_optim_params_list.append({'params': self.model.parameters(),
                                            'lr': 0.01,
                                            'momentum': 0.9,
                                            'weight_decay': 0.0005})
        
        model_optim_params_list.append({'params': self.classifer.parameters(),
                                            'lr': 0.1,
                                            'momentum': 0.9,
                                            'weight_decay': 0.0005})
        # init optimizer
        self.model_optimizer, self.model_optimizer_scheduler = self.init_optimizers(model_optim_params_list)
        
        # init criterion
        self.criterions['PerformanceLoss'] = nn.CrossEntropyLoss()
        self.criterions['FeatureLoss'] = create_loss(feat_dim=64, num_classes=100)
        self.criterion_weights['PerformanceLoss'] = 1.0
        self.criterion_weights['FeatureLoss'] = 0.01

        # init centroids
        self.criterions['FeatureLoss'].centroids.data = self.centroids_cal(train_data, device=device)
        optim_params = [{'params': self.criterions["FeatureLoss"].parameters(),
                        'lr': 0.01,
                        'momentum': 0.9,
                        'weight_decay': 0.0005}]
        self.criterion_optimizer, self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)


        train_data.dataset.target = train_data.dataset.target.astype(np.int64)


        for epoch in range(args.epochs):
            torch.cuda.empty_cache()
            # Iterate over dataset
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # inputs, labels = inputs.to(self.device), labels.to(self.device)
                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(x, labels, 
                                       centroids=True,
                                       phase='train')
                    self.batch_loss(labels)
                    self.batch_backward()

            self.model_optimizer_scheduler.step()
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()


    def train_init(self, device, cls_num_list=None, alpha=None):
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                         weight_decay=self.args.wd, amsgrad=True)

        if "focal" in self.args.method:
            # alpha = torch.Tensor([i/sum(cls_num_list) if i != 0 else 1/sum(cls_num_list) for i in cls_num_list])
            criterion = FocalLoss(gamma=0.5)
        elif "cbloss" in self.args.method:
            if "cifar10_lt" in self.args.dataset:
                beta = 0.999999
                gama = 1.0
            elif "cifar100_lt" in self.args.dataset:
                beta = 0.99
                gama = 0.8
            criterion = CB_loss(cls_num_list, self.class_num, device, beta=beta, gamma=gama)
        elif "lade" in self.args.method:
            criterion_perform = PriorCELoss(num_classes=self.class_num, prior_txt=cls_num_list).to(device)
            criterion_routeweight = LADELoss(num_classes=self.class_num, prior_txt=cls_num_list, remine_lambda=0.01).to(device)
            criterion_mdcs = MDCSLoss(cls_num_list=cls_num_list, tau=2).to(device)
            criterion = {"perform": criterion_perform, "routeweight": criterion_routeweight, "reverse": criterion_mdcs}
        elif "ldam" in self.args.method:
            criterion = LDAMLoss(cls_num_list=cls_num_list, reweight_ep=self.args.comm_round * 2/3).to(device)
        elif "contrast" in self.args.method:
            criterion = MDCSLoss(cls_num_list=cls_num_list, tau=2).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        return criterion, optimizer
        
    def class_count (self, data):
        labels = np.array(data.dataset.target)
        class_data_num = []
        for l in np.unique(labels):
            class_data_num.append(len(labels[labels == l]))
        return class_data_num

    def centroids_cal(self, data, device):

        centroids = torch.zeros(self.class_num, 128).cuda()

        print('Calculating centroids.')

        self.model.eval()
        self.classifer.eval()

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels in data: #tqdm(data):
                inputs, labels = inputs.to(device), labels.to(device)
                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]

        # Average summed features with class count
        centroids /= torch.tensor(data.dataset.get_cls_num_list()).float().unsqueeze(1).cuda()

        return centroids

    def test(self, test_data, round_idx, cls_num_list, device, args):
        model = self.model
        model.to(device)
        model.eval()

        if self.classifer is not None:
            classifer = self.classifer
            classifer.to(device)
            classifer.eval()

        # criterion2, _ = self.train_init(device, cls_num_list)


        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
        }
        test_data.dataset.target = test_data.dataset.target.astype(np.int64)
        criterion = nn.CrossEntropyLoss().to(device)
        extra_info = {}
        batch_loss = {"total_loss":0, "perform_loss":0, "routeweight_loss":0,"reverse_loss":0}
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                if self.args.contrast:
                    if isinstance(x, list):
                        x[0], x[1], labels = x[0].to(device), x[1].to(device), target.to(device)
                        data = torch.cat([x[0], x[1]], dim=0).cuda() 
                        target = torch.cat([labels, labels], dim=0).cuda()
                        
                        pred,logits = model(data)
                        pred1 = torch.chunk(pred, chunks=2, dim=0)[0].squeeze(0)
                        extra_info.update({
                            "logits": logits.transpose(0, 1),
                            "epoch": round_idx
                        })
                        # perform_loss = criterion2["perform"](pred1, labels)
                        # routeweight_loss = criterion2["routeweight"](pred1, labels)
                        # reverse_loss = criterion2["reverse"](output_logits=pred, target=target, extra_info=extra_info)
                        # loss = perform_loss + args.lade_weight * routeweight_loss + args.reverse_weight * reverse_loss
                        
                    else:
                        data = x.to(device)

                        target = target.to(device)
                        pred,logits = model(data)


                        # perform_loss = criterion2["perform"](pred, target)
                        # routeweight_loss = criterion2["routeweight"](pred, target)
                        pred2 = torch.cat([pred, pred], dim=0)
                        target2 = torch.cat([target, target], dim=0)
                        logits = torch.cat([logits, logits], dim=0)
                        # for debug loss
                        extra_info.update({
                            "logits": logits.transpose(0, 1),
                            "epoch": round_idx
                        })
                        # reverse_loss = criterion2["reverse"](output_logits=pred2, target=target2, extra_info=extra_info)
                        # loss = perform_loss + args.lade_weight * routeweight_loss + args.reverse_weight * reverse_loss
                else:
                    data = x.to(device)
                    target = target.to(device)
                    
                    if self.args.fintune:
                        self.batch_forward(data, target, 
                                        centroids=True,
                                        phase='test')
                    else:
                        pred, _ = model(data)

                if "lade" in self.args.method:
                    pred += torch.log(torch.ones(self.class_num)/self.class_num).to(device)
                if self.args.fintune:
                    loss = criterion(self.logits, target)
                    _, predicted = torch.max(self.logits, -1)
                else:
                    loss = criterion(pred, target)
                    _, predicted = torch.max(pred, -1)

                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

                if self.args.contrast:
                    batch_loss["total_loss"]+=loss.item()
                    batch_loss["perform_loss"]+=perform_loss.item()
                    batch_loss["routeweight_loss"]+=routeweight_loss.item()
                    batch_loss["reverse_loss"]+=reverse_loss.item()
        if self.args.contrast:
            batch_loss = {k:v/batch_idx for k,v in batch_loss.items()}
        
        return metrics, batch_loss



    def test_for_all_labels(self, test_data, device):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            'all_preds': 0,
            'Many acc': 0,
            'Medium acc': 0,
            'Few acc': 0,
        }

        label_smaple_num = {}
        for i in range(self.class_num):
            metrics[i] = 0
            label_smaple_num[i] = 0

        test_data.dataset.target = test_data.dataset.target.astype(np.int64)
        criterion = nn.CrossEntropyLoss().to(device)

        all_preds = torch.tensor([])
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                if self.args.contrast:
                    if isinstance(x, list):
                        x = x[0].to(device)
                    else:
                        x = x.to(device)
                    target = target.to(device)
                    pred,_ = model(x)
                else:
                    x = x.to(device)
                    target = target.to(device)
                    if self.args.fintune:
                        self.batch_forward(x, target, 
                                    centroids=True,
                                    phase='test')
                    else:
                        pred, _ = model(x)
                    
                if "lade" in self.args.method:
                    pred += torch.log(torch.ones(self.class_num)/self.class_num).to(device)
                if self.args.fintune:
                    loss = criterion(self.logits, target)
                    _, predicted = torch.max(self.logits, -1)
                else:
                    loss = criterion(pred, target)
                    _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

                all_preds = torch.cat((all_preds, predicted.cpu()), dim=0)

                for (idx, label) in enumerate(target):
                    if predicted[idx].eq(target[idx]):
                        metrics[label.item()] += 1
                    label_smaple_num[label.item()] += 1

            for label in range(self.class_num):
                if label_smaple_num[label] != 0:
                    metrics[label] = metrics[label] / label_smaple_num[label]

            if self.class_range is not None:
                for i in range(self.class_num):
                    if i < self.class_range[0]:
                        metrics['Many acc'] += metrics[i]
                    elif i < self.class_range[1]:
                        metrics['Medium acc'] += metrics[i]
                    else:
                        metrics['Few acc'] += metrics[i]

                metrics['Many acc'] /= self.class_range[0]
                metrics['Medium acc'] /= self.class_range[1] - self.class_range[0]
                if metrics['Medium acc'] < 0:
                    metrics['Medium acc'] = 0
                metrics['Few acc'] /= self.class_num - self.class_range[1]
                if metrics['Few acc'] < 0:
                    metrics['Few acc'] = 0

        metrics['all_preds'] = all_preds

        return metrics


def imshow(torch_batch_images, title = None):
    npimages = make_grid(torch_batch_images.detach().cpu())
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(npimages,(1, 2, 0)))
    plt.title(torch_batch_images.__str__ if title is None else title)
    plt.show()




