import abc
import torch
from repsim.geometry.length_space import LengthSpace, Point


# Typing hints: neural data of size (n, d)
NeuralData = torch.Tensor


class RepresentationMetricSpace(LengthSpace, abc.ABC):
    """Abstract base class for metrics between neural representations. Subclasses should use multiple-inheritance to
    subclass from both RepresentationMetricSpace and from a type of repsim.geometry.LengthSpace.
    """

    @abc.abstractmethod
    def neural_data_to_point(self, x: NeuralData) -> Point:
        """Convert (n,d) sized neural data into a point in the metric space, e.g. converting to an (n,n) sized RDM
        """
        pass

    @abc.abstractmethod
    def string_id(self) -> str:
        """Return a string description of this metric space object, e.g. for use as a dictionary key
        """
        pass
