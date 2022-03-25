from dassl.modeling import Backbone, BACKBONE_REGISTRY
from torchvision import models

class MyBackbone(Backbone):

    def __init__(self):
        super().__init__()
        # Create layers
        self.model = models.resnet18(pretrained=True)

    def forward(self, x):
        # Extract and return features
        return self.model(x)

@BACKBONE_REGISTRY.register()
def my_backbone(**kwargs):
    return MyBackbone()