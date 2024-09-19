from torch import nn

from fastreid.config import configurable

from .build import REID_CROWNS_REGISTRY


@REID_CROWNS_REGISTRY.register()
class DummyCrown(nn.Module):
    """
    A dummy crown that only changes the output format.
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
            'variance_vector': uncertainty["variance_vector"]
        }