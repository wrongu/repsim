import abc
import torch
from repsim.geometry.manifold import LengthSpace, Point


# Typing hints: neural data of size (m, d)
NeuralData = torch.Tensor


class RepresentationMetricSpace(LengthSpace, abc.ABC):
    """Abstract base class for metrics between neural representations.

    Subclasses should use multiple-inheritance to subclass from both RepresentationMetricSpace and
    from a type of repsim.geometry.LengthSpace.
    """

    @property
    def m(self):
        # By default, assume that self.m is the first index of self.shape. May be overridden if
        # need be.
        return self.shape[0]

    @m.setter
    def m(self, val):
        self.shape = (val,) + self.shape[1:]

    @abc.abstractmethod
    def neural_data_to_point(self, x: NeuralData) -> Point:
        """Convert (m,d) sized neural data into a point in the metric space, e.g. converting to an
        (m,m) sized RDM."""
        pass

    @abc.abstractmethod
    def string_id(self) -> str:
        """Return a string description of this metric space object, e.g. for use as a dictionary
        key."""
        pass

    @property
    @abc.abstractmethod
    def is_spherical(self) -> bool:
        """Return True if this metric's length() is in [0,pi] and may be interpreted as an arc
        length on some hypersphere."""
        pass
