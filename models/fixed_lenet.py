import torch
import torch.nn as nn


class FixedLeNet(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=32,
                 fixed_classifier_feat_dim=9):  # fixed_classifier_feat_dim=2 polygon2d
        super(FixedLeNet, self).__init__()

        # used in fixed classifier ##########
        self.l2_norm = False
        self.l2_scale = 1
        #####################################

        feat_map_sz = img_sz // 4
        self.n_feat = 50 * feat_map_sz * feat_map_sz

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
        )

        # the junction (500 is the number of activations before the fixed classifier)
        self.fc1 = nn.Linear(500, fixed_classifier_feat_dim)

        # the fixed classifier
        # TODO: NB WILL NOT BE USED WHEN INSTANTIATED BY AGENT (OVERWRITTEN BY MODULEDICT)
        self.last = nn.Linear(fixed_classifier_feat_dim, out_dim, bias=False)

    def features(self, x):

        # same as lenet ###
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        ###################

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


def fixedLeNetC(out_dim=10):  # LeNet with color input
    return FixedLeNet(out_dim=out_dim, in_channel=3, img_sz=32)
