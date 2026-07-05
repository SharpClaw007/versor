"""Versor — a language where the program is the path."""
from .asm import AsmError, assemble, assemble_path
from .builder import ProgramBuilder, arm, arm_seg
from .decode import DECODERS, get_decoder
from .errors import LoadError, VersorFault
from .interp import classify, lerp_programs
from .loader import Program, from_dict, load, save, to_dict
from .machine import Machine, RunResult, run_program
from .quat import Quat
from .trace import Trace

__version__ = "0.4.0"
__all__ = [
    "ProgramBuilder", "arm", "arm_seg", "LoadError", "VersorFault",
    "Program", "from_dict", "load", "save", "to_dict",
    "Machine", "RunResult", "run_program", "Quat", "Trace",
    "DECODERS", "get_decoder", "classify", "lerp_programs",
    "AsmError", "assemble", "assemble_path", "__version__",
]
