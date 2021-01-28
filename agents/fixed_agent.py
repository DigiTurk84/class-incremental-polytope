from __future__ import print_function
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from .exp_replay import Naive_Rehearsal


# ==============================================================================
# Four fixed classifiers geometry: polygon, hypercube, dsimplex, dorthoplex
def polygon2D(num_classes=10):
    import math
    feat_dim = 2

    unif_vec = np.zeros((feat_dim, num_classes))
    for n in range(0, num_classes):
        unif_vec[0, n] = 1 * math.cos(2 * math.pi * n / num_classes)
        unif_vec[1, n] = 1 * math.sin(2 * math.pi * n / num_classes)
    return unif_vec


# ------------------------------------------------------------------------------
def dcube(num_classes=10):
    import math
    def hypercube_vrtcs(n_dims=10, vertices=(-0.5, 0.5)):
        import itertools
        X = np.array(list(itertools.product(vertices, repeat=n_dims)))
        y = (np.sum(np.clip(X, a_min=0, a_max=1), axis=1) >= (n_dims / 2.0)).astype(np.int)
        return (X, y)

    feat_dim = math.ceil(math.log2(num_classes))
    a = hypercube_vrtcs(feat_dim)
    hc = a[0]
    hc = hc / np.linalg.norm(hc, axis=1, keepdims=True)  # OK UNIT NORMALIZED
    hc = hc.transpose()
    return hc


# ------------------------------------------------------------------------------
def dsimplex(num_classes=10):
    def simplex_coordinates2(m):
        # add the credit
        import numpy as np

        x = np.zeros([m, m + 1])
        for j in range(0, m):
            x[j, j] = 1.0

        a = (1.0 - np.sqrt(float(1 + m))) / float(m)

        for i in range(0, m):
            x[i, m] = a

        #  Adjust coordinates so the centroid is at zero.
        c = np.zeros(m)
        for i in range(0, m):
            s = 0.0
            for j in range(0, m + 1):
                s = s + x[i, j]
            c[i] = s / float(m + 1)

        for j in range(0, m + 1):
            for i in range(0, m):
                x[i, j] = x[i, j] - c[i]

        #  Scale so each column has norm 1. UNIT NORMALIZED
        s = 0.0
        for i in range(0, m):
            s = s + x[i, 0] ** 2
        s = np.sqrt(s)

        for j in range(0, m + 1):
            for i in range(0, m):
                x[i, j] = x[i, j] / s

        return x

    feat_dim = num_classes - 1
    ds = simplex_coordinates2(feat_dim)
    return ds


# ------------------------------------------------------------------------------
def dorthoplex(num_classes=10):
    feat_dim = np.ceil(num_classes / 2).astype(np.int)
    cp = np.identity(feat_dim)
    cp = np.vstack((cp, -cp)).transpose()
    return cp


# ==============================================================================


class FixedNaiveRehearsal(Naive_Rehearsal):

    def __init__(self,
                 agent_config,
                 fixed_classifier_feat_dim,
                 fixed_weights):
        self.fixed_classifier_feat_dim = fixed_classifier_feat_dim
        self.fixed_weights = fixed_weights
        super(FixedNaiveRehearsal, self).__init__(agent_config)

    def create_model(self):
        cfg = self.config

        # import pdb
        # pdb.set_trace()

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        # Simplex has feature dimension as: N_OUT - 1 size
        # used to allocate N_OUT virtual classes
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](
            fixed_classifier_feat_dim=self.fixed_classifier_feat_dim)  # added fixed classifier feat dimension

        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        # TODO: NB WILL OVERWRITE THE DEFAULT LAST LAYER OF EACH NETWORK
        for task, out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat, out_dim, bias=False)  # Remove bias here for fixed classifier
            model.last[task].weight.requires_grad = False  # set no gradient for the fixed classifier
            model.last[task].weight.copy_(self.fixed_weights)  # set the weights for the classifier

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model


class FixedSimplexNaiveRehearsal(FixedNaiveRehearsal):

    def __init__(self, agent_config):
        out_dim = agent_config['out_dim']['All']
        fixed_classifier_feat_dim = out_dim - 1
        fixed_weights = torch.from_numpy(dsimplex(num_classes=out_dim).transpose())
        super().__init__(agent_config, fixed_classifier_feat_dim, fixed_weights)


class FixedOrthoplexNaiveRehearsal(FixedNaiveRehearsal):

    def __init__(self, agent_config):
        out_dim = agent_config['out_dim']['All']
        fixed_classifier_feat_dim = int(np.ceil(out_dim / 2).astype(int))
        fixed_weights = torch.from_numpy(dorthoplex(num_classes=out_dim).transpose())
        super().__init__(agent_config, fixed_classifier_feat_dim, fixed_weights)


class FixedPolygonal2DNaiveRehearsal(FixedNaiveRehearsal):
    def __init__(self, agent_config):
        out_dim = agent_config['out_dim']['All']
        fixed_classifier_feat_dim = 2
        fixed_weights = torch.from_numpy(polygon2D(num_classes=out_dim).transpose())
        super().__init__(agent_config, fixed_classifier_feat_dim, fixed_weights)


# actual instantiation by command line

def Fixed_Simplex_Naive_Rehearsal_100(agent_config):
    agent = FixedSimplexNaiveRehearsal(agent_config)
    agent.memory_size = 100
    return agent

def Fixed_Simplex_Naive_Rehearsal_200(agent_config):
    agent = FixedSimplexNaiveRehearsal(agent_config)
    agent.memory_size = 200
    return agent

def Fixed_Simplex_Naive_Rehearsal_400(agent_config):
    agent = FixedSimplexNaiveRehearsal(agent_config)
    agent.memory_size = 400
    return agent

def Fixed_Simplex_Naive_Rehearsal_800(agent_config):
    agent = FixedSimplexNaiveRehearsal(agent_config)
    agent.memory_size = 800
    return agent


def Fixed_Simplex_Naive_Rehearsal_1100(agent_config):
    agent = FixedSimplexNaiveRehearsal(agent_config)
    agent.memory_size = 1100
    return agent

def Fixed_Simplex_Naive_Rehearsal_1400(agent_config):
    agent = FixedSimplexNaiveRehearsal(agent_config)
    agent.memory_size = 1400
    return agent


def Fixed_Simplex_Naive_Rehearsal_4400(agent_config):
    agent = FixedSimplexNaiveRehearsal(agent_config)
    agent.memory_size = 4400
    return agent


def Fixed_Simplex_Naive_Rehearsal_5600(agent_config):
    agent = FixedSimplexNaiveRehearsal(agent_config)
    agent.memory_size = 5600
    return agent
