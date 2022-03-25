from torch.nn import functional as F

from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.modeling import build_network
from dassl.engine.trainer import SimpleNet

from torchvision import models

@TRAINER_REGISTRY.register()
class ResNet18(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        cfg = self.cfg

        self.model = models.resnet18(pretrained=True)
        self.model.to(self.device)
        print("# params: {:,}".format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(
            self.optim, cfg.OPTIM
        )
        self.register_model(
            "model", self.model, self.optim, self.sched
        )

    def forward_backward(self, batch):
        input, label, domain = self.parse_batch_train(batch)
        out = self.model(input)
        loss = 0
        loss += F.cross_entropy(out, label)

        self.model_backward_and_update(loss, "model")
        loss_summary = {"loss":loss.item()}

        if(self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def model_inference(self, input):
        return self.model(input)
