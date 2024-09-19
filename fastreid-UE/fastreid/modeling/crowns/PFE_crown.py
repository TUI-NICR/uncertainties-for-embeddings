from torch import nn

from fastreid.config import configurable

from .build import REID_CROWNS_REGISTRY


@REID_CROWNS_REGISTRY.register()
class PFE_crown(nn.Module):
    """
    A dummy crown that only changes the output format.
    WARNING: this was abandoned before being tested and is probably unnecessary.
    """

    @configurable
    def __init__(
        self
    ) -> None:
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    @classmethod
    def from_config(cls, cfg):
        return {}

    def forward(self, features, uncertainty):
        return {
            'logits': None,
            'mean_vector': features["mean_vector"],
            'variance_vector': uncertainty["variance_vector"].clamp(1e-8)
        }