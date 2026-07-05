"""VHL — a tiny high-level language that compiles to Versor chains.

The interesting compiler problem in Versor is that storage is geometric:
registers are four vectors, memory is position-addressed, and every
instruction moves the machine. VHL v1 stays on the easy side of that line —
values live in registers only, so the allocator is a pool, not a path
planner (spilling to spatial memory *is* path planning and is the v2
problem). Consequences:

- R0 is reserved for the unit vector (repeat's decrement).
- Three registers (R1-R3) hold everything: named variables, expression
  temporaries, and loop counters. Exceeding them is a CompileError, not a
  spill.
- ``a * b`` needs a constant side (SCALE takes an immediate); variable *
  variable needs a loop — write it in VHL with ``repeat``.

Syntax (one statement per line, ``#`` comments):

    let x = 5
    print x * 2 + 1
    repeat x {
        print x
        let x = x - 1
    }

``repeat n`` runs its body ceil(n) times for positive n and zero times for
n <= 0 (the countdown branch exits on the zero-accumulator tie rule).
Scalars are carried in the frame-local x slot; the compiler emits no frame
rotations, so world and frame coordinates coincide throughout.
"""
from __future__ import annotations

import re

from .builder import ProgramBuilder, arm
from .errors import LoadError

NUM_RE = r"\d+(?:\.\d+)?"
TOKEN_RE = re.compile(rf"\s*({NUM_RE}|[A-Za-z_]\w*|[()+\-*={{}}])")
KEYWORDS = {"let", "print", "repeat"}


class CompileError(LoadError):
    pass


def _err(ln: int, msg: str) -> CompileError:
    return CompileError(f"line {ln}: {msg}")


def _tokenize(text: str, ln: int) -> list[str]:
    out, pos = [], 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise _err(ln, f"cannot tokenize {text[pos:].strip()!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


class _ExprParser:
    """Recursive descent with constant folding.

    expr   := term (('+' | '-') term)*
    term   := factor ('*' factor)*     -- at least one side must fold const
    factor := NUM | NAME | '(' expr ')' | '-' factor
    """

    def __init__(self, tokens: list[str], ln: int):
        self.t = tokens
        self.i = 0
        self.ln = ln

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self):
        node = self.expr()
        if self.peek() is not None:
            raise _err(self.ln, f"unexpected {self.peek()!r} after expression")
        return node

    def expr(self):
        node = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            rhs = self.term()
            if node[0] == "num" and rhs[0] == "num":
                node = ("num", node[1] + rhs[1] if op == "+"
                        else node[1] - rhs[1])
            else:
                node = ("add" if op == "+" else "sub", node, rhs)
        return node

    def term(self):
        node = self.factor()
        while self.peek() == "*":
            self.take()
            rhs = self.factor()
            if node[0] == "num" and rhs[0] == "num":
                node = ("num", node[1] * rhs[1])
            elif rhs[0] == "num":
                node = ("mul", node, rhs[1])
            elif node[0] == "num":
                node = ("mul", rhs, node[1])
            else:
                # variable * variable: MULR, available in the Versor-32
                # dialects — the compiler auto-selects icosa32 when used
                node = ("mulvar", node, rhs)
        return node

    def factor(self):
        tok = self.take()
        if tok is None:
            raise _err(self.ln, "unexpected end of expression")
        if tok == "(":
            node = self.expr()
            if self.take() != ")":
                raise _err(self.ln, "missing ')'")
            return node
        if tok == "-":
            node = self.factor()
            if node[0] == "num":
                return ("num", -node[1])
            return ("neg", node)
        if re.fullmatch(NUM_RE, tok):
            return ("num", float(tok))
        if tok == "input" and self.peek() == "(":
            self.take()
            if self.take() != ")":
                raise _err(self.ln, "input takes no arguments: input()")
            return ("inp",)
        if re.fullmatch(r"[A-Za-z_]\w*", tok) and tok not in KEYWORDS:
            return ("var", tok)
        raise _err(self.ln, f"unexpected token {tok!r} in expression")


class _Compiler:
    def __init__(self, decoder: str = "cubic26"):
        self.b = ProgramBuilder("vhl", decoder=decoder)
        self.c = self.b.chain("compiled from VHL")
        self.vars: dict[str, int] = {}
        self.free = [3, 2, 1]
        self.owners: dict[int, str] = {}
        self.label_n = 0

    def alloc(self, what: str, ln: int) -> int:
        if not self.free:
            held = ", ".join(f"R{r} = {w}" for r, w in sorted(self.owners.items()))
            raise _err(ln, f"out of registers allocating {what} "
                           f"(3 available; held: {held}). VHL v1 does not "
                           f"spill to spatial memory")
        r = self.free.pop()
        self.owners[r] = what
        return r

    def release(self, r: int):
        del self.owners[r]
        self.free.append(r)

    def fresh(self, kind: str) -> str:
        self.label_n += 1
        return f"{kind}{self.label_n}"

    # --- expression codegen: value ends up in A's frame-local x slot ---

    def zero_a(self, ln: int):
        t = self.alloc("zero-temp", ln)
        self.c.movr(t).sub(t)  # A - A = 0, whatever A held
        self.release(t)

    def eval(self, node, ln: int):
        kind = node[0]
        if kind == "num":
            v = node[1]
            if v > 1e-9:
                self.c.loadi(v)
            elif v < -1e-9:
                self.c.loadi(-v)
                self.negate_a(ln)
            else:
                self.zero_a(ln)
        elif kind == "var":
            name = node[1]
            if name not in self.vars:
                raise _err(ln, f"undefined variable {name!r}")
            self.c.mova(self.vars[name])
        elif kind == "add":
            # a variable operand is already in a register: no temp needed
            for a, b in ((node[1], node[2]), (node[2], node[1])):
                if b[0] == "var" and b[1] in self.vars:
                    self.eval(a, ln)
                    self.c.add(self.vars[b[1]])
                    return
            t = self.alloc("add-temp", ln)
            self.eval(node[1], ln)
            self.c.movr(t)
            self.eval(node[2], ln)
            self.c.add(t)
            self.release(t)
        elif kind == "sub":
            if node[2][0] == "var" and node[2][1] in self.vars:
                self.eval(node[1], ln)
                self.c.sub(self.vars[node[2][1]])
                return
            t = self.alloc("sub-temp", ln)
            self.eval(node[2], ln)
            self.c.movr(t)
            self.eval(node[1], ln)
            self.c.sub(t)
            self.release(t)
        elif kind == "mul":
            k = node[2]
            self.eval(node[1], ln)
            if k > 1e-9:
                self.c.scale(k)
            elif k < -1e-9:
                self.c.scale(-k)
                self.negate_a(ln)
            else:
                self.zero_a(ln)
        elif kind == "neg":
            self.eval(node[1], ln)
            self.negate_a(ln)
        elif kind == "inp":
            self.c.inp()
        elif kind == "mulvar":
            a, bnode = node[1], node[2]
            if bnode[0] == "var" and bnode[1] in self.vars:
                self.eval(a, ln)
                self.c.mulr(self.vars[bnode[1]])
            elif a[0] == "var" and a[1] in self.vars:
                self.eval(bnode, ln)
                self.c.mulr(self.vars[a[1]])
            else:
                t = self.alloc("mul-temp", ln)
                self.eval(bnode, ln)
                self.c.movr(t)
                self.eval(a, ln)
                self.c.mulr(t)
                self.release(t)
        else:  # pragma: no cover
            raise _err(ln, f"internal: unknown node {kind}")

    def negate_a(self, ln: int):
        t = self.alloc("neg-temp", ln)
        self.c.movr(t).sub(t).sub(t)  # 0, then -A
        self.release(t)

    # --- statements ---

    def stmt_let(self, name: str, node, ln: int):
        self.eval(node, ln)
        if name not in self.vars:
            self.vars[name] = self.alloc(f"var {name}", ln)
        self.c.movr(self.vars[name])

    def stmt_print(self, node, ln: int):
        self.eval(node, ln)
        self.c.out()

    def loop_begin(self, node, ln: int):
        self.eval(node, ln)
        counter = self.alloc("repeat counter", ln)
        self.c.movr(counter)
        top, body, end = (self.fresh("top"), self.fresh("body"),
                          self.fresh("end"))
        self.c.label(top)
        self.c.mova(counter)
        self.c.branch(
            arm("NOP", 1.0, guard=(-1, 0, 0), to=end),   # first: 0-tie exits
            arm("NOP", 1.0, guard=(1, 0, 0), to=body),
        )
        self.c.at(body)
        return counter, top, end

    def loop_end(self, counter: int, top: str, end: str):
        self.c.mova(counter).sub(0).movr(counter)
        self.c.op("NOP", 1.0, to=top)
        self.c.at(end)
        self.release(counter)


def _uses_extended(node) -> bool:
    if node[0] in ("inp", "mulvar"):
        return True
    return any(isinstance(child, tuple) and _uses_extended(child)
               for child in node[1:])


def _parse_stmts(src: str) -> list[tuple]:
    stmts = []
    for ln, raw in enumerate(src.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = _tokenize(line, ln)
        if tokens == ["}"]:
            stmts.append(("close", ln))
            continue
        head = tokens[0]
        if head == "let":
            if len(tokens) < 4 or tokens[2] != "=" or not re.fullmatch(
                    r"[A-Za-z_]\w*", tokens[1]) or tokens[1] in KEYWORDS:
                raise _err(ln, "expected: let NAME = expr")
            stmts.append(("let", tokens[1],
                          _ExprParser(tokens[3:], ln).parse(), ln))
        elif head == "print":
            stmts.append(("print", _ExprParser(tokens[1:], ln).parse(), ln))
        elif head == "repeat":
            if tokens[-1] != "{":
                raise _err(ln, "expected: repeat expr {")
            stmts.append(("repeat", _ExprParser(tokens[1:-1], ln).parse(), ln))
        else:
            raise _err(ln, f"unknown statement {head!r} "
                           "(statements: let, print, repeat)")
    return stmts


def compile_vhl(src: str) -> ProgramBuilder:
    stmts = _parse_stmts(src)
    # input() and var*var need the Versor-32 extended opcodes: compile
    # against icosa32 when they appear, cubic26 otherwise
    extended = any(_uses_extended(s[-2]) for s in stmts
                   if s[0] in ("let", "print", "repeat"))
    comp = _Compiler(decoder="icosa32" if extended else "cubic26")
    comp.c.loadi(1).movr(0)  # R0 = unit, reserved for repeat decrements
    stack: list[tuple[int, str, str]] = []  # open repeat blocks

    for stmt in stmts:
        if stmt[0] == "close":
            if not stack:
                raise _err(stmt[1], "unmatched '}'")
            comp.loop_end(*stack.pop())
        elif stmt[0] == "let":
            comp.stmt_let(stmt[1], stmt[2], stmt[3])
        elif stmt[0] == "print":
            comp.stmt_print(stmt[1], stmt[2])
        else:
            stack.append(comp.loop_begin(stmt[1], stmt[2]))

    if stack:
        raise CompileError("unclosed '{' at end of file")
    comp.c.halt()
    return comp.b


def compile_path(path: str) -> ProgramBuilder:
    with open(path) as f:
        return compile_vhl(f.read())
