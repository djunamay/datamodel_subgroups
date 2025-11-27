class CounterfactualOutputs:
    """A property-only container for arbitrary keyword arguments."""
    def __init__(self, **kwargs):
        # store all keyword arguments as attributes
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"