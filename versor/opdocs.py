"""Per-mnemonic documentation, shared by the LSP hover and docs tooling.

Each entry: (signature, effect). `n` is the segment magnitude (the
operand); `idx` means floor(n) mod 4; frame-local means conjugated through
the live frame F.
"""

OP_DOCS: dict[str, tuple[str, str]] = {
    # face directions: data
    "LOADI": ("LOADI n",
              "A = n in the frame-local x slot (A = F·(n,0,0)·F⁻¹)."),
    "STORE": ("STORE [n=1]",
              "Write A to the arrival cell: M[cell(P)] = A."),
    "LOAD": ("LOAD [n=1]  (n ≥ 2 is EXEC)",
             "A = M[cell(P)] at the arrival cell. With n ≥ 2 this is EXEC: "
             "execute the cell's stored vector as a full instruction, "
             "movement included (chained EXECs walk stored code)."),
    "MOVR": ("MOVR r0..r3 | n", "R[idx] = A."),
    "MOVA": ("MOVA r0..r3 | n", "A = R[idx]."),
    "HALT": ("HALT [n=1]",
             "Halt; the root chain's net displacement is the result."),
    # edge directions: arithmetic & frame
    "ADD": ("ADD r0..r3 | n", "A = A + R[idx]."),
    "SUB": ("SUB r0..r3 | n", "A = A − R[idx]."),
    "SCALE": ("SCALE n", "A = A · n (n > 0; sign changes need SUB)."),
    "DOT": ("DOT r0..r3 | n",
            "A = A·R[idx] as a scalar in the frame-local x slot."),
    "CROSS": ("CROSS r0..r3 | n", "A = A × R[idx]."),
    "NORM": ("NORM [n=1]", "A = A/|A|; faults DivisionByZero on |A| < ε."),
    "PROJ": ("PROJ r0..r3 | n", "A = projection of A onto R[idx]."),
    "REJ": ("REJ r0..r3 | n", "A = A − proj(A, R[idx])."),
    "ROTF": ("ROTF angle",
             "Rotate the frame about its own x-axis by `angle` radians "
             "(intrinsic; pi-expressions allowed). Reinterprets every "
             "downstream direction."),
    "ROTG": ("ROTG angle", "Rotate the frame about its own y-axis."),
    "ROTH": ("ROTH angle", "Rotate the frame about its own z-axis."),
    "OUT": ("OUT [n=1]  (OUTC = OUT 2)",
            "Append frame-local A.x to the output; n ≥ 2 emits chr(round)."),
    # corner directions: control
    "CALL": ("CALL chain [scale]",
             "Call chain floor(n) under the live frame — orientation is the "
             "argument. frac(n) is the Sim(3) scale: s' = s·2^(2·frac−1); "
             "one call carries a factor in [0.5, 2)."),
    "RET": ("RET [n=1]",
            "Return; caller's A = callee net displacement; frame and scale "
            "restored, position kept. Chain end is an implicit RET."),
    "JMPZ": ("JMPZ [n=1]",
             "If |A| < ε, skip the next segment (it still moves)."),
    "JMPP": ("JMPP [n=1]",
             "If frame-local A.x > ε, skip the next segment (still moves)."),
    "PUSHF": ("PUSHF [n=1]", "Push (F, P, s) onto the aux stack."),
    "POPF": ("POPF [n=1]",
             "Pop the aux stack, restoring frame and scale — NOT position."),
    "NOP": ("NOP [n=1]", "Move only."),
    "FAULT": ("FAULT [code=1]", "Deliberate fault; the operand is the code."),
    # extended Versor-32 (icosa32 / sphere32 only)
    "INP": ("INP [n=1]  — Versor-32",
            "Read the next input scalar into the frame-local x slot; faults "
            "InputExhausted when the buffer is empty."),
    "SWAP": ("SWAP r0..r3 | n  — Versor-32", "Swap A and R[idx]."),
    "PUSHA": ("PUSHA [n=1]  — Versor-32", "Push A onto the data stack."),
    "POPA": ("POPA [n=1]  — Versor-32",
             "Pop the data stack into A; faults StackUnderflow when empty."),
    "MULR": ("MULR r0..r3 | n  — Versor-32",
             "A = A · (frame-local x of R[idx]) — variable × variable."),
    "LOADP": ("LOADP [n=1]  — Versor-32",
              "A = P (read-only position introspection)."),
    # assembler pseudo-ops
    "OUTC": ("OUTC", "OUT with n = 2: emit chr(round(frame-local A.x))."),
    "EXEC": ("EXEC [n=2]",
             "LOAD with n ≥ 2: execute the arrival cell's stored vector."),
    "BR": ("BR guard: OP [arg] -> target, ...",
           "Branch: take the arm whose guard (frame-local, rotated by the "
           "live frame) best matches Â; first-listed wins ties, and a zero "
           "accumulator ties everything — list the exit arm first."),
    "SEG": ("SEG (x, y, z)",
            "Emit an explicit frame-local segment (authoring-frame aware)."),
    "SEGRAW": ("SEGRAW (x, y, z)",
               "Emit a raw world-space segment, bypassing the authoring frame."),
    "OP": ("OP MNEMONIC n",
           "Escape hatch: any mnemonic with an explicit raw magnitude."),
}

DIRECTIVE_DOCS: dict[str, tuple[str, str]] = {
    ".name": (".name text", "Program name."),
    ".decoder": (".decoder cubic26|icosa32|sphere26|sphere32",
                 "Direction decoder; extended Versor-32 opcodes need "
                 "icosa32 or sphere32."),
    ".chain": (".chain [name]",
               "Start a chain; the first is the entry point. Named chains "
               "are CALL targets."),
}
