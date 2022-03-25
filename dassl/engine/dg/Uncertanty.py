from torch.nn import functional as F

from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.modeling import build_network
from dassl.engine.trainer import SimpleNet

# 想要将这些类在算法运行之前就加入到注册表中，
# 需要修改本文件夹下的__init__.py文件，将对应的类名导入
@TRAINER_REGISTRY.register()
class Uncertanty(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = "cuda"

    def build_model(self):
        cfg = self.cfg

        print("Building Feature Extractor")
        self.featureExtractor = build_network(cfg.TRAINER.MyExp.G_ARCH, verbose=cfg.VERBOSE)
        self.featureExtractor.to(self.device)
        print("# params: {:,}".format(count_num_param(self.featureExtractor)))
        self.optim_F = build_optimizer(self.featureExtractor, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("featureExtractor",
            self.featureExtractor, self.optim_F, self.sched_F)

        print("Building discriminator")
        self.dis = SimpleNet(cfg, cfg.Model, self.num_source_domains)
        self.dis.to(self.device)
        print("# params: {:,}".format(count_num_param(self.dis)))
        self.optim_cls = build_optimizer(self.dis, cfg.OPTIM)
        self.sched_cls = build_lr_scheduler(
            self.optim_cls, cfg.OPTIM
        )
        self.register_model(
            "dis", self.dis, self.optim_cls, self.sched_cls
        )

        print("")


    def forward_backward(self, batch):
        input, label, domain = self.parse_batch_train(batch)
        out = self.featureExtractor(input)
        loss_g = 0
        loss_g += F.cross_entropy(out, label)
        self.model_backward_and_update(loss_g, "model")

        loss_summary = {"loss_g":loss_g.item()}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary
    
    def model_inference(self, input):
        return self.featureExtractor(input)

@TRAINER_REGISTRY.register()
class Split(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.optim = None
        self.device = "cuda" 

    def build_model(self):
        cfg = self.cfg
        self.model = build_network(cfg.TRAINER.MyExp.SPLITNET, verbose=cfg.VERBOSE)
        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

    def forward_backward(self, batch):
        input, label, domain = self.parse_batch_train(batch)
        out1, out2 = self.model(input, mode="train")
        
        loss1 = F.cross_entropy(out1, label)
        loss2 = F.cross_entropy(out2, label)
        
        loss_g = 0
        if self.epoch <= self.max_epoch//2:
            loss_g += loss1
        else:
            loss_g += (loss1 - loss2)

        self.model_backward_and_update(loss_g, "model")
        loss_summary = {"loss_g":loss_g.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def model_inference(self, input):
        return self.model(input, mode="self-test")
