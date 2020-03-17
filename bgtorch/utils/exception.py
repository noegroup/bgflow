

class BGTorchException(Exception):
    """Base class for all BGTorch exceptions."""
    pass


class SoftDependencyException(BGTorchException):
    """A soft dependency is not installed."""
    def __init__(self, dependency):
        super(SoftDependencyException, self).__init__(f"{dependency} not installed.")

