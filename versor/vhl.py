"""VHL — a tiny high-level language that compiles to Versor chains.

The interesting compiler problems in Versor are geometric: storage is four
registers plus position-addressed memory, and every instruction moves the
machine. VHL keeps values in registers (spilling to spatial memory is the
path-planning problem, versor/route.py territory); the allocator is a
3-slot pool with R0 reserved as the unit vector.

v3 language:

    fn square(x) {              # functions compile to chains; recursion
        return x * x            # works (the call stack is the machine's)
    }
    let n = input()
    while n {                   # while / if test: expr > 0
        if n - 3 {
            print square(n)
        } else {
            print 0 - n
        }
        let n = n - 1
    }
    repeat 2 { print 100 }      # repeat: counted loop sugar

Calling convention (data stack, so any of fn/input()/var*var selects the
icosa32 dialect): the caller PUSHAs its live registers, then the arguments
left-to-right, and CALLs; the callee POPAs parameters right-to-left into
its own registers and PUSHAs its result (falling off the end returns 0);
the caller POPAs the result and restores its registers. Parking the result
during restoration needs one free register — with three general registers,
a call with live locals can exhaust them (a CompileError, not a spill).
"""
from __future__ import annotations

import re

from .builder import ProgramBuilder, arm
from .errors import LoadError

NUM_RE = r"\d+(?:\.\d+)?"
TOKEN_RE = re.compile(rf"\s*({NUM_RE}|[A-Za-z_]\w*|[(),+\-*={{}}])")
KEYWORDS = {"let", "print", "repeat", "while", "if", "else", "fn", "return",
            "input"}


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
    """expr := term (('+'|'-') term)* ; term := factor ('*' factor)* ;
    factor := NUM | NAME | NAME(args) | input() | '(' expr ')' | '-' factor
    Constant folding on numeric subtrees."""

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
                node = ("mulvar", node, rhs)  # MULR (Versor-32)
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
            if self.peek() == "(":
                self.take()
                args = []
                if self.peek() != ")":
                    while True:
                        args.append(self.expr())
                        nxt = self.take()
                        if nxt == ")":
                            break
                        if nxt != ",":
                            raise _err(self.ln, "expected ',' or ')' in call")
                else:
                    self.take()
                return ("call", tok, args)
            return ("var", tok)
        raise _err(self.ln, f"unexpected token {tok!r} in expression")


# ---------- parsing to statement lists ----------

def _parse_units(src: str):
    """Split source into main statements + {fn name: (params, stmts, ln)}.
    Statements: (kind, ...); block structure stays flat with begin/close."""
    main: list[tuple] = []
    fns: dict[str, tuple] = {}
    target = main            # where statements currently go
    fn_depth = 0             # brace depth inside a fn body
    block_depth = 0          # brace depth in the current target

    for ln, raw in enumerate(src.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = _tokenize(line, ln)

        if tokens[0] == "fn":
            if target is not main:
                raise _err(ln, "nested fn is not allowed")
            if block_depth:
                raise _err(ln, "fn must be declared at top level")
            if (len(tokens) < 5 or tokens[2] != "(" or tokens[-1] != "{"
                    or tokens[-2] != ")"):
                raise _err(ln, "expected: fn NAME(a, b, ...) {")
            name = tokens[1]
            if name in KEYWORDS or not re.fullmatch(r"[A-Za-z_]\w*", name):
                raise _err(ln, f"bad function name {name!r}")
            if name in fns:
                raise _err(ln, f"duplicate function {name!r}")
            params = [t for t in tokens[3:-2] if t != ","]
            if any(not re.fullmatch(r"[A-Za-z_]\w*", p) or p in KEYWORDS
                   for p in params):
                raise _err(ln, "parameters must be plain names")
            if len(params) != len(set(params)):
                raise _err(ln, "duplicate parameter names")
            body: list[tuple] = []
            fns[name] = (params, body, ln)
            target = body
            fn_depth = 1
            continue

        if tokens == ["}"] and target is not main and fn_depth == 1 \
                and block_depth == 0:
            target = main
            fn_depth = 0
            continue

        kind = _classify(tokens, ln, in_fn=target is not main)
        target.append(kind)
        if kind[0] in ("repeat", "while", "if"):
            block_depth += 1
        elif kind[0] == "close":
            if block_depth == 0 and target is not main:
                # shouldn't happen (handled above), defensive
                raise _err(ln, "unmatched '}'")
            block_depth = max(0, block_depth - 1)
        # 'else' closes one block and opens another: depth unchanged

    if target is not main:
        raise CompileError("unclosed fn body at end of file")
    return main, fns


def _classify(tokens: list[str], ln: int, in_fn: bool) -> tuple:
    if tokens == ["}"]:
        return ("close", ln)
    if tokens[0] == "}" and tokens[1:] == ["else", "{"]:
        return ("else", ln)
    head = tokens[0]
    if head == "let":
        if len(tokens) < 4 or tokens[2] != "=" or not re.fullmatch(
                r"[A-Za-z_]\w*", tokens[1]) or tokens[1] in KEYWORDS:
            raise _err(ln, "expected: let NAME = expr")
        return ("let", tokens[1], _ExprParser(tokens[3:], ln).parse(), ln)
    if head == "print":
        return ("print", _ExprParser(tokens[1:], ln).parse(), ln)
    if head == "return":
        if not in_fn:
            raise _err(ln, "return outside a function")
        node = (_ExprParser(tokens[1:], ln).parse() if len(tokens) > 1
                else ("num", 0.0))
        return ("return", node, ln)
    if head in ("repeat", "while", "if"):
        if tokens[-1] != "{":
            raise _err(ln, f"expected: {head} expr {{")
        return (head, _ExprParser(tokens[1:-1], ln).parse(), ln)
    raise _err(ln, f"unknown statement {head!r} "
                   "(statements: let, print, repeat, while, if, fn, return)")


def _uses_extended(node) -> bool:
    if node[0] in ("inp", "mulvar", "call"):
        return True
    return any(isinstance(c, tuple) and _uses_extended(c) for c in node[1:])


def _stmts_use_extended(stmts) -> bool:
    for s in stmts:
        for part in s[1:-1]:
            if isinstance(part, tuple) and _uses_extended(part):
                return True
    return False


# ---------- code generation ----------

class _ChainCompiler:
    def __init__(self, chain, fn_ids: dict[str, tuple[int, int]]):
        self.c = chain
        self.fn_ids = fn_ids  # name -> (chain id, arity)
        self.vars: dict[str, int] = {}
        self.free = [3, 2, 1]
        self.owners: dict[int, str] = {}
        self.label_n = 0

    def alloc(self, what: str, ln: int) -> int:
        if not self.free:
            held = ", ".join(f"R{r} = {w}" for r, w in sorted(self.owners.items()))
            raise _err(ln, f"out of registers allocating {what} "
                           f"(3 available; held: {held}). Spilling to "
                           "spatial memory is not implemented")
        r = self.free.pop()
        self.owners[r] = what
        return r

    def release(self, r: int):
        del self.owners[r]
        self.free.append(r)

    def fresh(self, kind: str) -> str:
        self.label_n += 1
        return f"{kind}{self.label_n}"

    # --- expressions: value ends in the frame-local x slot of A ---

    def zero_a(self, ln: int):
        t = self.alloc("zero-temp", ln)
        self.c.movr(t).sub(t)
        self.release(t)

    def negate_a(self, ln: int):
        t = self.alloc("neg-temp", ln)
        self.c.movr(t).sub(t).sub(t)
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
            a, b = node[1], node[2]
            if b[0] == "var" and b[1] in self.vars:
                self.eval(a, ln)
                self.c.mulr(self.vars[b[1]])
            elif a[0] == "var" and a[1] in self.vars:
                self.eval(b, ln)
                self.c.mulr(self.vars[a[1]])
            else:
                t = self.alloc("mul-temp", ln)
                self.eval(b, ln)
                self.c.movr(t)
                self.eval(a, ln)
                self.c.mulr(t)
                self.release(t)
        elif kind == "call":
            self.eval_call(node[1], node[2], ln)
        else:  # pragma: no cover
            raise _err(ln, f"internal: unknown node {kind}")

    def eval_call(self, name: str, args: list, ln: int):
        if name not in self.fn_ids:
            raise _err(ln, f"undefined function {name!r}")
        cid, arity = self.fn_ids[name]
        if len(args) != arity:
            raise _err(ln, f"{name} takes {arity} argument(s), got {len(args)}")
        live = sorted(self.owners)  # registers to survive the call
        for r in live:
            self.c.mova(r).pusha()
        for a in args:
            self.eval(a, ln)
            self.c.pusha()
        self.c.call(cid)
        if live:
            park = self.alloc("call-result", ln)
            self.c.popa().movr(park)
            for r in reversed(live):
                self.c.popa().movr(r)
            self.c.mova(park)
            self.release(park)
        else:
            self.c.popa()

    # --- statements ---

    def stmt_let(self, name: str, node, ln: int):
        self.eval(node, ln)
        if name not in self.vars:
            self.vars[name] = self.alloc(f"var {name}", ln)
        self.c.movr(self.vars[name])

    def branch_on_a(self, pos_label: str, nonpos_label: str):
        """expr > 0 goes to pos_label; zero ties exit to nonpos_label."""
        self.c.branch(
            arm("NOP", 1.0, guard=(-1, 0, 0), to=nonpos_label),
            arm("NOP", 1.0, guard=(1, 0, 0), to=pos_label),
        )

    def run(self, stmts, *, fn_exit: str | None = None):
        stack: list[tuple] = []
        for stmt in stmts:
            kind = stmt[0]
            if kind == "let":
                self.stmt_let(stmt[1], stmt[2], stmt[3])
            elif kind == "print":
                self.eval(stmt[1], stmt[2])
                self.c.out()
            elif kind == "return":
                self.eval(stmt[1], stmt[2])
                self.c.pusha()
                self.c.op("NOP", 1.0, to=fn_exit)
                # unreachable continuation vertex for any following code
                self.c.at(self.fresh("dead"))
            elif kind == "repeat":
                node, ln = stmt[1], stmt[2]
                self.eval(node, ln)
                counter = self.alloc("repeat counter", ln)
                self.c.movr(counter)
                top, body, end = (self.fresh("rt"), self.fresh("rb"),
                                  self.fresh("re"))
                self.c.label(top)
                self.c.mova(counter)
                self.branch_on_a(body, end)
                self.c.at(body)
                stack.append(("repeat", counter, top, end))
            elif kind == "while":
                node, ln = stmt[1], stmt[2]
                top, body, end = (self.fresh("wt"), self.fresh("wb"),
                                  self.fresh("we"))
                self.c.label(top)
                self.eval(node, ln)
                self.branch_on_a(body, end)
                self.c.at(body)
                stack.append(("while", top, end))
            elif kind == "if":
                node, ln = stmt[1], stmt[2]
                then, other, end = (self.fresh("it"), self.fresh("ie"),
                                    self.fresh("ix"))
                self.eval(node, ln)
                self.branch_on_a(then, other)
                self.c.at(then)
                stack.append(("if", other, end, False))
            elif kind == "else":
                if not stack or stack[-1][0] != "if":
                    raise _err(stmt[1], "'} else {' without a matching if")
                _, other, end, seen = stack.pop()
                if seen:
                    raise _err(stmt[1], "duplicate else")
                self.c.op("NOP", 1.0, to=end)
                self.c.at(other)
                stack.append(("if", other, end, True))
            elif kind == "close":
                if not stack:
                    raise _err(stmt[1], "unmatched '}'")
                block = stack.pop()
                if block[0] == "repeat":
                    _, counter, top, end = block
                    self.c.mova(counter).sub(0).movr(counter)
                    self.c.op("NOP", 1.0, to=top)
                    self.c.at(end)
                    self.release(counter)
                elif block[0] == "while":
                    _, top, end = block
                    self.c.op("NOP", 1.0, to=top)
                    self.c.at(end)
                else:  # if
                    _, other, end, seen = block
                    self.c.op("NOP", 1.0, to=end)
                    if not seen:
                        self.c.at(other)
                        self.c.op("NOP", 1.0, to=end)
                    self.c.at(end)
        if stack:
            raise CompileError("unclosed '{' at end of file")


def compile_vhl(src: str) -> ProgramBuilder:
    main_stmts, fns = _parse_units(src)
    extended = (bool(fns) or _stmts_use_extended(main_stmts)
                or any(_stmts_use_extended(body) for _, body, _ in fns.values()))
    b = ProgramBuilder("vhl", decoder="icosa32" if extended else "cubic26")

    main_chain = b.chain("main")
    fn_ids = {name: (1 + i, len(params))
              for i, (name, (params, _b, _ln)) in enumerate(fns.items())}

    mc = _ChainCompiler(main_chain, fn_ids)
    mc.c.loadi(1).movr(0)  # R0 = unit, reserved (shared by all chains)
    mc.run(main_stmts)
    mc.c.halt()

    for name, (params, body, decl_ln) in fns.items():
        chain = b.chain(f"fn {name}")
        fc = _ChainCompiler(chain, fn_ids)
        exit_label = "fnexit"
        # parameters arrive on the data stack, last argument on top
        for p in reversed(params):
            fc.vars[p] = fc.alloc(f"param {p}", decl_ln)
            fc.c.popa().movr(fc.vars[p])
        fc.run(body, fn_exit=exit_label)
        if not body or body[-1][0] != "return":
            fc.zero_a(decl_ln)  # fell off the end: return 0
            fc.c.pusha()
            fc.c.op("NOP", 1.0, to=exit_label)
        fc.c.at(exit_label)
        fc.c.ret()

    return b


def compile_path(path: str) -> ProgramBuilder:
    with open(path) as f:
        return compile_vhl(f.read())
