import torch
import torch.nn as nn


class FixedMLP(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, fixed_classifier_feat_dim=None):
        super().__init__()

        # used in fixed classifier ##########
        self.l2_norm = False
        self.l2_scale = 1
        #####################################


        self.in_dim = in_channel * img_sz * img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

        # the junction (500 is the number of activations before the fixed classifier)
        self.fc1 = nn.Linear(hidden_dim, fixed_classifier_feat_dim)

        # the fixed classifier
        # TODO: NB WILL NOT BE USED WHEN INSTANTIATED BY AGENT (OVERWRITTEN BY MODULEDICT)
        self.last = nn.Linear(fixed_classifier_feat_dim, out_dim, bias=False)

    def features(self, x):
        x = self.linear(x.view(-1, self.in_dim))

        # used in fixed classifier ##########
        x = self.fc1(x)  # added for our approach. Comment for the original class

        if self.l2_norm:
            # apply normalization to features (weights are already normalized from init)
            norm = x.norm(p=2, dim=1, keepdim=True) + 1e-5
            x = x.div(norm) * self.l2_scale
            # TODO: NB IT RETURN THE NORMALIZED FEATURES WE LOSE THE ORIGINAL ONES IF THEY ARE NEEDED
        ######################################

        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def FixedMLP100(**kwargs):
    return FixedMLP(hidden_dim=100, **kwargs)


def FixedMLP400(**kwargs):
    return FixedMLP(hidden_dim=400, **kwargs)


def FixedMLP1000(**kwargs):
    return FixedMLP(hidden_dim=1000, **kwargs)


def FixedMLP2000(**kwargs):
    return FixedMLP(hidden_dim=2000, **kwargs)


def FixedMLP5000(**kwargs):
    return FixedMLP(hidden_dim=5000, **kwargs)
