"""Load the R3M ResNet18 backbone from ``r3m/model.pt``.

The checkpoint is a plain torchvision-style ResNet18 ``state_dict`` with the
final ``fc`` layer removed (120 keys, 512-dim output). It still contains the
full conv trunk: ``conv1/bn1`` + ``layer1 .. layer4``.

Usage
-----
Full backbone (512-dim global feature)::

    from r3m.load_r3m_backbone import load_r3m_backbone
    net = load_r3m_backbone()          # resnet18 with R3M weights, fc=Identity
    feat = net(x)                      # x: (B, 3, 224, 224) -> (B, 512)

Truncated trunk (conv1 .. layer2), mirroring the del-based pattern::

    from r3m.load_r3m_backbone import R3MTrunk
    trunk = R3MTrunk()
    feat = trunk(x)                    # (B, 3, 224, 224) -> (B, 128, 28, 28)

Preprocessing (R3M expects ImageNet-normalized RGB, 224x224)::

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),                       # -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
"""
import os

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

_DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")


def load_r3m_backbone(ckpt_path=_DEFAULT_CKPT, truncate=False):
    """Build a torchvision ResNet18 and load the R3M weights into it.

    Args:
        ckpt_path: path to the stripped R3M ``state_dict`` (default: sibling
            ``model.pt``).
        truncate: if ``True``, drop ``layer3``/``layer4``/``fc`` after loading,
            mirroring the original::

                self.cnn = resnet18(weights="DEFAULT")
                del self.cnn.layer3
                del self.cnn.layer4
                del self.cnn.fc

            The delete happens *after* ``load_state_dict`` on purpose, so the
            strict load still sees the full trunk and matches exactly.
            NOTE: once truncated, the default ``resnet`` ``forward`` no longer
            works (it references the deleted layers) -- use :class:`R3MTrunk`,
            which provides a matching ``forward``.

    Returns:
        nn.Module: the ``resnet18`` with R3M weights loaded (``eval`` mode).
    """
    net = resnet18()          # random init -- do NOT pass weights= (avoids download)
    net.fc = nn.Identity()    # R3M drops the classification head

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(sd, strict=True)   # loads the FULL trunk (incl. layer3/4)

    if truncate:
        del net.layer3
        del net.layer4
        del net.fc

    net.eval()
    return net


class R3MTrunk(nn.Module):
    """ResNet18 truncated at ``layer2`` (conv1 .. layer2), R3M-initialized.

    Mirrors the ``del layer3 / layer4 / fc`` pattern but keeps a working
    ``forward``. For a 224x224 input the output is ``(B, 128, 28, 28)``.
    """

    def __init__(self, ckpt_path=_DEFAULT_CKPT):
        super().__init__()
        self.cnn = load_r3m_backbone(ckpt_path, truncate=True)

    def forward(self, x):
        c = self.cnn
        x = c.conv1(x)
        x = c.bn1(x)
        x = c.relu(x)
        x = c.maxpool(x)
        x = c.layer1(x)
        x = c.layer2(x)
        return x


if __name__ == "__main__":
    dummy = torch.zeros(1, 3, 224, 224)

    full = load_r3m_backbone()
    with torch.no_grad():
        print("full   :", tuple(full(dummy).shape))          # (1, 512)

    trunk = R3MTrunk()
    with torch.no_grad():
        print("trunc  :", tuple(trunk(dummy).shape))         # (1, 128, 28, 28)
