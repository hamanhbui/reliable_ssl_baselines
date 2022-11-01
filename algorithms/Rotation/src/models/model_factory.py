from algorithms.Rotation.src.models.mnistnet import MNIST_CNN
from algorithms.Rotation.src.models.cifar10net import CIFAR10_CNN
from algorithms.Rotation.src.models.resnet import Wide_ResNet


nets_map = {"mnistnet": MNIST_CNN, "cifar10net": CIFAR10_CNN, "wide_resnet": Wide_ResNet}


def get_model(name):
    if name not in nets_map:
        raise ValueError("Name of model unknown %s" % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn
