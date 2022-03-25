import torch
from torch.nn import functional as F
import torch.nn as nn
from torchvision import models

from .build import NETWORK_REGISTRY


# Feature extractor
class resnetFeatureExtractor(nn.Module):
    def __init__(self):
        super(resnetFeatureExtractor, self).__init__()
        self.model = models.resnet18(pretrained=True)

    def fletch(self):
        return self.model.fc.in_features

    def forward(self, x, mode = "split"):
        if mode == "split":
            for block in list(self.model.children()):
                if type(block) != nn.Linear:
                    x = block(x)
            return x
        elif mode == "uncertanty":
            for block in list(self.model.children()):
                if type(block) != nn.Linear and type(block) != nn.AdaptiveAvgPool2d:
                    x = block(x)
            return x


class bottleNet(nn.Module):
    def __init__(self, in_dim, bottle_net_dim=256):
        super(bottleNet, self).__init__()
        self.bottleNet = nn.Linear(in_dim, bottle_net_dim)

    def forward(self, x):
        x = self.bottleNet(x)
        return x

# classification network
class CLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.flatten = nn.Flatten(1, 3)
        self.softmax = nn.Softmax(dim=-1)
        self.main = nn.Sequential(self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        # out = [x]
        out = self.flatten(x)
        # print("*******************")
        # print(out.shape)
        # # print(torch.chunk(x, 2, dim=2)[0].shape)
        # print("*******************")
        out = self.fc(out)
        return self.softmax(out)

        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


class featureSplitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.featureExtractor = resnetFeatureExtractor()
        self.classifier_train = CLS(self.featureExtractor.fletch() // 2, 7)
        self.classifier_eval = CLS(self.featureExtractor.fletch(), 7)

    def splitFeature(self, feature, chunks=2, dim=2):
        (x1, x2) = torch.chunk(feature, chunks, dim=dim)
        return x1, x2

    def forward(self, x, mode="train", glob=False):
        """
        mode = train/test/self-test
        """
        x = self.featureExtractor(x)
        x1, x2 = self.splitFeature(x)

        # feature = x2

        if glob:  # add global average pooling
            c = F.adaptive_avg_pool3d(x, (1, 1, 1))
            c = c.expand_as(x1)
            x1 = torch.cat([x1, c], 1)
            x2 = torch.cat([x2, c], 1)
            x1 = self.classifier_eval(x1)
            x2 = self.classifier_eval(x2)
        else:
            x1 = self.classifier_train(x1)
            x2 = self.classifier_train(x2)

        if mode == "train":
            return x1, x2
        elif mode == "test":
            return self.classifier_eval(x)
        elif mode == "self-test":
            return x1

def uncertantyModeling(feature):
    mu = torch.mean(feature, (2, 3))
    sigma = torch.var(feature, (2, 3))

    Sigma_mu = torch.var(mu, dim=0)
    Sigma_sigma = torch.var(sigma, dim=0)

    epsilon_mu = torch.randn(Sigma_mu.shape)
    epsilon_sigma = torch.randn(Sigma_sigma.shape)

    beta = mu + torch.mul(epsilon_mu, Sigma_mu)
    gamma = sigma + torch.mul(epsilon_sigma, Sigma_sigma)

    sigma = sigma.unsqueeze(2).unsqueeze(3)
    mu = mu.unsqueeze(2).unsqueeze(3)
    gamma = gamma.unsqueeze(2).unsqueeze(3)
    beta = beta.unsqueeze(2).unsqueeze(3)

    return torch.mul(gamma, (feature - mu)/sigma) + beta

class UncertantyNet(nn.Module):
    def __init__(self):
        super().__init__(UncertantyNet, self)
        self.featureExtractor = resnetFeatureExtractor()
        self.fc_in = self.featureExtractor.fletch()
        # self.classifier = CLS(self.fc_in, 7)
        # self.discriminator = CLS(self.fc_in, 7)
        self.bottle = bottleNet(self.fc_in * 2, self.fc_in)
        self.adaAvePooling = F.adaptive_avg_pool2d()

    def forward(self, x):
        x = self.featureExtractor(x, mode="uncertanty")
        x1 = uncertantyModeling(x)
        x2 = uncertantyModeling(x)
        x = torch.cat((x1, x2), dim=1)
        x = F.adaptive_avg_pool2d(x, (1, 1)) # B 1026 1 1
        x = self.bottle(x)  # B 512 1 1
        return x




@NETWORK_REGISTRY.register()
def FeatureSplitNet(**kwargs):
    net = models.resnet18(pretrained=True)
    return net

@NETWORK_REGISTRY.register()
def SplitNet(**kwargs):
    SpNet = featureSplitNet()
    return SpNet

@NETWORK_REGISTRY.register()
def uncertantyNet(**kwargs):
    net = uncertantyNet()
    return net
