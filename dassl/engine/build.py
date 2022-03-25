'''Registry:提供名字到对象的映射'''
from dassl.utils import Registry, check_availability

TRAINER_REGISTRY = Registry("TRAINER")


def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()   # 目标的键
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:# Verbose表示是否输出日志
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)

'''
应该是获取到通过名称获取到训练器，然后将参数传入到训练器中，这里默认的训练器是Vanlilla2
'''