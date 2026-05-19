"""Custom exceptions for the roundtable core module."""


class DimensionMismatchError(ValueError):
    """Raised when two embedding vectors have incompatible dimensions.

    This is a subclass of ValueError so that generic ValueError handlers
    still catch it, but callers can also catch it specifically to apply
    targeted fallback logic.
    """

    def __init__(self, dim1: int, dim2: int) -> None:
        self.dim1 = dim1
        self.dim2 = dim2
        super().__init__(
            f"Embedding dimension mismatch: {dim1} vs {dim2} – "
            f"similar vectors cannot be compared."
        )
