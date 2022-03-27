import torch
from torch.nn import functional as F
import torch.nn as nn
from torchvision import models

from .build import NETWORK_REGISTRY
import random


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
        self.Flatten = nn.Flatten(1,3)
        self.bottleNet = nn.Linear(in_dim, bottle_net_dim)

    def forward(self, x):
        x = self.Flatten(x)
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

        out = self.fc(out)
        return self.softmax(out)

class featureSplitNet(nn.Module):
    def __init__(
        self, 
        FE_mode = "uncertanty", 
        domain_num = 7,
        num_chunk_dim = 2
    ):
        super().__init__()
        self.FE_mode = FE_mode  # 特征提取器的模式
        self.domain_num = domain_num
        self.num_chunk_dim = num_chunk_dim
        self.featureExtractor = resnetFeatureExtractor()
        # self.classifier_train = CLS(self.featureExtractor.fletch() // 2, 7)
        # self.classifier_eval = CLS(self.featureExtractor.fletch(), 7)
        self.cls_label = CLS(self.featureExtractor.fletch(), 7)
        self.cls_domain = CLS(self.featureExtractor.fletch(), domain_num)
        self.adaptiveAvePool = nn.AdaptiveAvgPool2d((1,1))

    def splitFeature(self, feature, chunks= 2,dim=3):# B C H W
        (x1, x2) = torch.chunk(feature, chunks, dim=dim)
        return x1, x2

    def forward(self, x, mode="train", glob=False, chunks=2):
        """
        mode = train/test/self-test
        """
        x = self.featureExtractor(x, mode = self.FE_mode)

        # 将空间维度合并后再进行划分
        if self.num_chunk_dim == 2:
            B,C,H,W = x.size()
            x_view = x.view(B, C, -1)
            x1, x2 = self.splitFeature(x_view, dim=-1)
            x1 = torch.unsqueeze(x1, 3)
            x2 = torch.unsqueeze(x2, 3)
        else:
            x1, x2 = self.splitFeature(x, chunks)


        # Adaptive Average Pooling
        if self.FE_mode == "uncertanty":
            x1 = self.adaptiveAvePool(x1)
            x2 = self.adaptiveAvePool(x2)
            x = self.adaptiveAvePool(x)

        if glob:  # add global average pooling
            c = F.adaptive_avg_pool3d(x, (1, 1, 1))
            c = c.expand_as(x1)
            x1 = torch.cat([x1, c], 1)
            x2 = torch.cat([x2, c], 1)
            x1 = self.classifier_eval(x1)
            x2 = self.classifier_eval(x2)
        else:
            x1 = self.cls_label(x1)
            x2 = self.cls_domain(x2)

        if mode == "train":
            return x1, x2
        elif mode == "test":
            return self.cls_label(x)
        elif mode == "self-test":
            return (x1 + x2)/2

class UncertaintyNet(nn.Module):
    def __init__(
        self, num_classes = 7
    ):
        super().__init__()
        self.featureExtractor = resnetFeatureExtractor()
        self.fc_in = self.featureExtractor.fletch()
        # self.classifier = CLS(self.fc_in, 7)
        # self.discriminator = CLS(self.fc_in, 7)
        self.bottle = bottleNet(self.fc_in * 2, self.fc_in)
        # self.adaAvePooling = F.adaptive_avg_pool2d
        self.prob = 0.5
        self.cls = CLS(self.fc_in, num_classes)

    def uncertaintyModeling(self, feature, eps=1e-6):
        mu = torch.mean(feature, (2, 3))
        sigma = torch.var(feature, (2, 3))

        Sigma_mu = torch.var(mu, dim=0)
        Sigma_sigma = torch.var(sigma, dim=0)

        epsilon_mu = torch.randn(Sigma_mu.shape)
        epsilon_sigma = torch.randn(Sigma_sigma.shape)

        # 让所有的变量都在GPU中
        epsilon_mu = epsilon_mu.cuda()
        epsilon_sigma = epsilon_sigma.cuda()

        beta = mu + torch.mul(epsilon_mu, Sigma_mu)
        gamma = sigma + torch.mul(epsilon_sigma, Sigma_sigma)

        sigma = sigma.unsqueeze(2).unsqueeze(3)
        mu = mu.unsqueeze(2).unsqueeze(3)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        return torch.mul(gamma, (feature - mu)/(sigma+eps)) + beta

    def forward(self, x):
        self.prob = 1
        x = self.featureExtractor(x, mode="uncertanty") # B 512 7 7

        # if random.random() <= self.prob:
        if 1:
            x1 = self.uncertaintyModeling(x)
            x2 = self.uncertaintyModeling(x)
        else:
            x1 = x
            x2 = x

        x = torch.cat((x1, x2), dim=1)
        x = F.adaptive_avg_pool2d(x, (1, 1)) # B 1026 1 1
        x = self.bottle(x)  # B 512

        # x = torch.squeeze(x) # B 512
        B, C = x.shape
        x = x.view(B, C, 1, 1)
        x = self.cls(x)
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
def Uncertainty(**kwargs):
    net = UncertaintyNet()
    return net

