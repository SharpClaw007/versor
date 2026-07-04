"""Runtime faults for the Versor machine."""
from __future__ import annotations


class VersorFault(Exception):
    """A runtime fault. `kind` is a stable machine-readable tag."""

    def __init__(self, kind: str, message: str, *, step: int | None = None,
                 chain: int | None = None, vertex: int | None = None):
        self.kind = kind
        self.message = message
        self.step = step
        self.chain = chain
        self.vertex = vertex
        loc = ""
        if step is not None:
            loc = f" [step {step}, chain {chain}, vertex {vertex}]"
        super().__init__(f"{kind}: {message}{loc}")


class LoadError(Exception):
    """A .vsr file failed validation."""
