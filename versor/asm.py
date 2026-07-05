"""Text assembler: .vasm -> .vsr.

The assembler is a thin front-end over the builder, so it inherits authoring-
frame tracking (segments after ROTF/ROTG/ROTH are re-aimed automatically) and
per-decoder cone centers.

Syntax (semicolon or # comments; blank lines ignored):

    .name countdown          ; program name
    .decoder cubic26         ; optional: cubic26 (default) | icosa32

    .chain entry             ; first chain is the entry point (id 0)
            LOADI 1
            MOVR r0          ; register ops take r0..r3 (encoded idx + 0.5)
            MOVR 2.0         ; ...or a raw magnitude (here: idx 2)
            LOADI 5
    loop:   OUT              ; 'label:' names the vertex before the next op
            SUB r0
            BR -x: HALT -> end, +x: NOP -> loop
                             ; branch arms: guard: OP [arg] -> target
                             ; first arm listed wins guard ties
    .chain fork              ; further chains are callable: CALL fork | CALL 1
            BR (1,0,0): LOADI 0.6 -> a, (-1,0,0): LOADI 2.5 -> b

Details:

- Guards: shorthand +x -x +y -y +z -z or an explicit (gx,gy,gz) vector.
- Angles for ROTF/ROTG/ROTH: a float or [k]pi[/m] (pi, pi/2, 3pi/4, 2pi/24).
- No-operand mnemonics (HALT, NOP, RET, JMPZ, JMPP, PUSHF, POPF, NORM, STORE,
  LOAD, OUT, FAULT) default to magnitude 1.0; OUTC is OUT with n = 2 (char
  mode); EXEC is LOAD with n >= 2 (execute the arrival cell's stored vector).
  Any of them accepts an explicit magnitude.
- SEG (x,y,z) emits an explicit frame-local segment; SEGRAW bypasses the
  authoring frame.
- Any instruction may end with '-> label' to target an existing or future
  vertex (this is how cycles without guards are written).
- Branch targets that are never defined become dead-end vertices (chain end).
"""
from __future__ import annotations

import math
import re

import numpy as np

from .builder import Arm, ProgramBuilder, arm
from .errors import LoadError

GUARDS = {
    "x": (1, 0, 0), "+x": (1, 0, 0), "-x": (-1, 0, 0),
    "y": (0, 1, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
    "z": (0, 0, 1), "+z": (0, 0, 1), "-z": (0, 0, -1),
}

REG_OPS = {"MOVR", "MOVA", "ADD", "SUB", "DOT", "CROSS", "PROJ", "REJ"}
FLOAT_OPS = {"LOADI", "SCALE"}
ANGLE_OPS = {"ROTF", "ROTG", "ROTH"}
DEFAULT_N = {"HALT": 1.0, "NOP": 1.0, "RET": 1.0, "JMPZ": 1.0, "JMPP": 1.0,
             "PUSHF": 1.0, "POPF": 1.0, "NORM": 1.0, "STORE": 1.0,
             "LOAD": 1.0, "FAULT": 1.0, "OUT": 1.0, "OUTC": 2.0, "EXEC": 2.0}
PSEUDO = {"OUTC": "OUT", "EXEC": "LOAD"}  # magnitude-band aliases

_LABEL_RE = re.compile(r"^([A-Za-z_]\w*):\s*(.*)$")
_REG_RE = re.compile(r"^[rR]([0-3])$")
_PI_RE = re.compile(r"^(\d+\.?\d*)?\s*\*?\s*pi\s*(?:/\s*(\d+\.?\d*))?$", re.I)


class AsmError(LoadError):
    pass


def _err(ln: int, msg: str) -> AsmError:
    return AsmError(f"line {ln}: {msg}")


def _strip(line: str) -> str:
    for marker in (";", "#"):
        if marker in line:
            line = line.split(marker, 1)[0]
    return line.strip()


def _parse_num(s: str, ln: int, what: str) -> float:
    try:
        return float(s)
    except ValueError:
        raise _err(ln, f"{what}: expected a number, got {s!r}")


def _parse_angle(s: str, ln: int) -> float:
    m = _PI_RE.match(s.strip())
    if m:
        k = float(m.group(1)) if m.group(1) else 1.0
        d = float(m.group(2)) if m.group(2) else 1.0
        return k * math.pi / d
    return _parse_num(s, ln, "angle")


def _parse_vec(s: str, ln: int, what: str) -> np.ndarray:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        raise _err(ln, f"{what}: expected (x, y, z), got {s!r}")
    parts = s[1:-1].split(",")
    if len(parts) != 3:
        raise _err(ln, f"{what}: expected 3 components, got {len(parts)}")
    return np.array([_parse_num(p, ln, what) for p in parts])


def _parse_guard(s: str, ln: int) -> np.ndarray:
    s = s.strip()
    if s.lower() in GUARDS:
        return np.array(GUARDS[s.lower()], dtype=float)
    return _parse_vec(s, ln, "guard")


def _split_top(s: str, sep: str = ",") -> list[str]:
    """Split on sep outside parentheses (guards contain commas)."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _operand(mnemonic: str, args: str, ln: int,
             chain_ids: dict[str, int], n_chains: int) -> float:
    """Resolve an instruction's argument string to a magnitude."""
    args = args.strip()
    if mnemonic in REG_OPS:
        if not args:
            raise _err(ln, f"{mnemonic} needs a register (r0..r3) or magnitude")
        m = _REG_RE.match(args)
        if m:
            return int(m.group(1)) + 0.5
        if args[0] in "rR" and not args[0:2].isdigit():
            raise _err(ln, f"{mnemonic}: bad register {args!r} (valid: r0..r3)")
        return _parse_num(args, ln, mnemonic)
    if mnemonic in FLOAT_OPS:
        if not args:
            raise _err(ln, f"{mnemonic} needs a numeric operand")
        return _parse_num(args, ln, mnemonic)
    if mnemonic in ANGLE_OPS:
        if not args:
            raise _err(ln, f"{mnemonic} needs an angle")
        return _parse_angle(args, ln)
    if mnemonic == "CALL":
        if not args:
            raise _err(ln, "CALL needs a chain index or name")
        if args in chain_ids:
            cid = chain_ids[args]
        else:
            try:
                cid = int(args)
            except ValueError:
                raise _err(ln, f"CALL: unknown chain {args!r}")
        if not 0 <= cid < n_chains:
            raise _err(ln, f"CALL: chain {cid} out of range (0..{n_chains - 1})")
        return cid + 0.5
    if mnemonic in DEFAULT_N:
        n = _parse_num(args, ln, mnemonic) if args else DEFAULT_N[mnemonic]
        if mnemonic == "EXEC" and n < 2.0:
            raise _err(ln, f"EXEC magnitude must be >= 2 (got {n}); "
                           "below 2 it is a plain LOAD")
        return n
    raise _err(ln, f"unknown mnemonic {mnemonic!r}")


def _prescan(lines: list[str]) -> tuple[str, str, dict[str, int], int]:
    """First pass: program name, decoder, chain name -> id, chain count."""
    name, decoder = "", "cubic26"
    chain_ids: dict[str, int] = {}
    n_chains = 0
    saw_code = False
    for ln, raw in enumerate(lines, 1):
        line = _strip(raw)
        if not line:
            continue
        if line.startswith("."):
            parts = line.split(None, 1)
            directive, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
            if directive == ".name":
                name = rest.strip()
            elif directive == ".decoder":
                decoder = rest.strip()
            elif directive == ".chain":
                cname = rest.split()[0] if rest.split() else ""
                if cname:
                    if cname in chain_ids:
                        raise _err(ln, f"duplicate chain name {cname!r}")
                    chain_ids[cname] = n_chains
                n_chains += 1
            else:
                raise _err(ln, f"unknown directive {directive!r}")
        else:
            if n_chains == 0 and not saw_code:
                n_chains = 1  # implicit entry chain
            saw_code = True
    if n_chains == 0:
        raise AsmError("no instructions found")
    return name, decoder, chain_ids, n_chains


def assemble(text: str) -> ProgramBuilder:
    lines = text.splitlines()
    name, decoder, chain_ids, n_chains = _prescan(lines)
    try:
        pb = ProgramBuilder(name, decoder=decoder)
    except ValueError as e:
        raise AsmError(str(e))
    c = None

    for ln, raw in enumerate(lines, 1):
        line = _strip(raw)
        if not line:
            continue

        if line.startswith("."):
            parts = line.split(None, 1)
            if parts[0].lower() == ".chain":
                c = pb.chain(parts[1].strip() if len(parts) > 1 else "")
            continue  # .name/.decoder handled in prescan

        if c is None:
            c = pb.chain("entry")  # implicit chain 0

        m = _LABEL_RE.match(line)
        if m:
            label, line = m.group(1), m.group(2).strip()
            try:
                if c._cursor is None:
                    c.at(label)      # continue at a (possibly future) vertex
                else:
                    c.label(label)   # name the current vertex
            except LoadError as e:
                raise _err(ln, str(e))
            if not line:
                continue

        parts = line.split(None, 1)
        mnemonic, args = parts[0].upper(), (parts[1] if len(parts) > 1 else "")

        try:
            if mnemonic == "BR":
                c.branch(*_parse_arms(args, ln, chain_ids, n_chains))
                continue

            to = None
            if "->" in args:
                args, target = args.rsplit("->", 1)
                args, to = args.strip(), target.strip()

            if mnemonic in ("SEG", "SEGRAW"):
                vec = _parse_vec(args, ln, mnemonic)
                (c.seg if mnemonic == "SEG" else c.seg_raw)(vec, to=to)
                continue
            if mnemonic == "OP":
                op_parts = args.split()
                if len(op_parts) != 2:
                    raise _err(ln, "OP needs: OP MNEMONIC magnitude")
                c.op(op_parts[0].upper(), _parse_num(op_parts[1], ln, "OP"), to=to)
                continue

            n = _operand(mnemonic, args, ln, chain_ids, n_chains)
            real = PSEUDO.get(mnemonic, mnemonic)
            if real in ANGLE_OPS:
                # route through the rot helpers so the authoring frame updates
                getattr(c, real.lower())(n)
                if to is not None:
                    raise _err(ln, f"{real}: '->' jump not supported on rotations")
            else:
                c.op(real, n, to=to)
        except AsmError:
            raise
        except LoadError as e:
            raise _err(ln, str(e))

    return pb


def _parse_arms(args: str, ln: int, chain_ids: dict[str, int],
                n_chains: int) -> list[Arm]:
    arms = []
    for spec in _split_top(args):
        if "->" not in spec:
            raise _err(ln, f"branch arm needs '-> target': {spec!r}")
        left, target = spec.rsplit("->", 1)
        if ":" not in left:
            raise _err(ln, f"branch arm needs 'guard: OP': {spec!r}")
        guard_s, op_s = left.split(":", 1)
        guard = _parse_guard(guard_s, ln)
        op_parts = op_s.split(None, 1)
        if not op_parts:
            raise _err(ln, f"branch arm missing opcode: {spec!r}")
        mnemonic = op_parts[0].upper()
        op_args = op_parts[1] if len(op_parts) > 1 else ""
        n = _operand(mnemonic, op_args, ln, chain_ids, n_chains)
        real = PSEUDO.get(mnemonic, mnemonic)
        try:
            arms.append(arm(real, n, guard=guard, to=target.strip()))
        except LoadError as e:
            raise _err(ln, str(e))
    return arms


def assemble_path(path: str) -> ProgramBuilder:
    with open(path) as f:
        return assemble(f.read())
