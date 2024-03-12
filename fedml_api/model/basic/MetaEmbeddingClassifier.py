import torch
import torch.nn as nn
from .CosNormClassifier import CosNorm_Classifier
# from utils import *

import pdb

class MetaEmbedding_Classifier(nn.Module):
    
    def __init__(self, feat_dim=2048, num_classes=1000):
        super(MetaEmbedding_Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)
        
    def forward(self, x, centroids, *args):
        
        # storing direct feature
        direct_feature = x.clone()

        batch_size = x.size(0)
        feat_size = x.size(1)
        
        # set up visual memory
        x_expand = x.clone().unsqueeze(1).expand(-1, self.num_classes, -1)
        centroids_expand = centroids.clone().unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids.clone() 
        
        # computing reachability
        dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
        values_nn, labels_nn = torch.sort(dist_cur, 1) # min to max
        scale = 10.0
        reachability = (scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)

        # computing memory feature by querying and associating visual memory
        values_memory = self.fc_hallucinator(x.clone())
        values_memory = values_memory.softmax(dim=1) # into probability distribution
        memory_feature = torch.matmul(values_memory, keys_memory)

        # computing concept selector
        concept_selector = self.fc_selector(x.clone())
        concept_selector = concept_selector.tanh() 
        x = reachability * (direct_feature + concept_selector * memory_feature)

        # storing infused feature
        infused_feature = concept_selector * memory_feature
        
        logits = self.cosnorm_classifier(x)

        return logits, [direct_feature, infused_feature]
    

def init_weights(model, weights_path, caffe=False, classifier=False):  
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))    
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
    else:      
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k] 
                   for k in model.state_dict()}
    model.load_state_dict(weights)   
    return model

def create_classifer(feat_dim=64, num_classes=100, stage1_weights=False, test=False, *args):
    print('Loading Meta Embedding Classifier.')
    clf = MetaEmbedding_Classifier(feat_dim, num_classes)

    if not test:
        if stage1_weights:
            clf.fc_hallucinator = init_weights(model=clf.fc_hallucinator,
                                                    weights_path='/root/OpenLongTailRecognition-OLTR/logs/%s/stage1/final_model_checkpoint.pth' % dataset,
                                                    classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf
