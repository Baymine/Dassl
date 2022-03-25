from dassl.utils import Registry, check_availability

# 一个档案馆，输入网络名称得到对应网络的
HEAD_REGISTRY = Registry("HEAD")


def build_head(name, verbose=True, **kwargs):
    avai_heads = HEAD_REGISTRY.registered_names()  # 载入档案中存在的网络名称
    check_availability(name, avai_heads)
    if verbose:
        print("Head: {}".format(name))
    return HEAD_REGISTRY.get(name)(**kwargs)  # 获取网络对象，并传入网络参数
