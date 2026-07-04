"""Versor — a language where the program is the path."""
from .builder import ProgramBuilder, arm, arm_seg
from .errors import LoadError, VersorFault
from .loader import Program, from_dict, load, save, to_dict
from .machine import Machine, RunResult, run_program
from .quat import Quat
from .trace import Trace

__version__ = "0.1.0"
__all__ = [
    "ProgramBuilder", "arm", "arm_seg", "LoadError", "VersorFault",
    "Program", "from_dict", "load", "save", "to_dict",
    "Machine", "RunResult", "run_program", "Quat", "Trace", "__version__",
]
