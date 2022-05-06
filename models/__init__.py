import torchvision
from models.e2_wide_resnet import e2wrn28_10R, e2wrn28_7R
from models.e2_resnet import E2ResNet, E2BasicBlock, E2Bottleneck

models = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
    "resnext50_32x4d": torchvision.models.resnext50_32x4d,
    "resnext101_32x8d": torchvision.models.resnext101_32x8d,
    "wide_resnet50_2": torchvision.models.wide_resnet50_2,
    "wide_resnet101_2": torchvision.models.wide_resnet101_2,
    "densenet121": torchvision.models.densenet121,
    "densenet": torchvision.models.DenseNet,
    # my e2 implementations
    "e2_resnet18": E2ResNet,
    # e2cnn_exp implementations
    "e2_wide_resnet28_10R": e2wrn28_10R,
    "e2_wide_resnet28_7R": e2wrn28_7R
}


def get_model(model_type, num_classes, args):
    model_args = {"num_classes": num_classes}
    # densenet_args = {"growth_rate": args.growth_rate,
    #                  "block_config": (3, 3, 3),
    #                  "num_init_features": args.num_init_features}
    e2_resnet = {"block": E2BasicBlock, "layers": [2, 2, 2, 2], "fixparams": False,
                 "f": False, # not equivariant to flips
                 "r": -1,  # no layer restriction
                 "conv2triv": True} # use group pooling}

    if model_type == "densenet":
        pass
        # model_args = {**model_args, **densenet_args}
    if model_type == "e2_resnet18":
        model_args = {**model_args, **e2_resnet}
    print("resnet18")
    return models[model_type](**model_args)

