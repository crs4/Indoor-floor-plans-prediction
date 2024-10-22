# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

from .nadirfloornet import build_nadirfloor


def build_model(args, train=True):
    return build_nadirfloor(args, train)

