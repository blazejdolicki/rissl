import torchvision

# temporary hack to import rissl
# TODO: Improve it once the final directory structure is determined
import sys
sys.path.insert(0, "/home/b.dolicki/thesis/")

from rissl.models import e2_resnet, resnet
import torchvision

models = {
    "resnet18": resnet.resnet18,
    "resnet34": resnet.resnet34,
    "resnet50": resnet.resnet50,
    "resnet101": resnet.resnet101,
    "resnet152": resnet.resnet152,
    "resnext50_32x4d": resnet.resnext50_32x4d,
    "resnext101_32x8d": resnet.resnext101_32x8d,
    "wide_resnet50_2": resnet.wide_resnet50_2,
    "wide_resnet101_2": resnet.wide_resnet101_2,
    "densenet121": torchvision.models.densenet121,
    "densenet": torchvision.models.DenseNet,
    # my equivariant implementations
    "e2_resnet18": e2_resnet.e2_resnet18,
    "e2_resnet50": e2_resnet.e2_resnet50,
    "e2_resnext50": e2_resnet.e2_resnext50,
}


def get_model(model_type, num_classes, args):
    model_args = {"num_classes": num_classes, "headless": args.headless}

    if model_type == "densenet":
        densenet_args = {"growth_rate": args.growth_rate,
                         "block_config": (3, 3, 3),
                         "num_init_features": args.num_init_features}
        model_args = {**model_args, **densenet_args}
    if is_equivariant(model_type):
        equivariant_args = {"N": args.N,
                            "F": args.F,
                            "sigma": args.sigma,
                            "restrict": args.restrict,
                            "flip": args.flip,
                            "fixparams": args.fixparams,
                            "deltaorth": args.deltaorth,
                            "last_hid_dims": args.last_hid_dims}
        model_args = {**model_args, **equivariant_args}

    return models[model_type](**model_args)

def is_equivariant(model_type):
    return "e2" in model_type