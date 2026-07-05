"""Language server for .vasm and .vhl.

The protocol layer (pygls) is only imported inside `main()`, so the pure
analysis functions here are importable and testable without the `lsp`
extra installed. Start with:  versor lsp   (stdio transport).

Editor wiring: the VS Code extension in tooling/vscode-versor launches it
automatically; any LSP-capable editor can run the command directly, e.g.
Zed/Helix/Neovim with `command = "versor", args = ["lsp"]` for the
`vasm` and `vhl` languages.
"""
from __future__ import annotations

import re

from .asm import DEFAULT_N, GUARDS, REG_OPS, assemble
from .errors import LoadError
from .isa import MNEMONIC_TO_TRIPLE
from .opdocs import DIRECTIVE_DOCS, OP_DOCS
from .vhl import compile_vhl

_LINE_RE = re.compile(r"^line (\d+): (.*)$", re.S)
_LABEL_DEF_RE = re.compile(r"^([A-Za-z_]\w*):")
_CHAIN_DEF_RE = re.compile(r"^\.chain\s+(\w+)")
_WORD_RE = re.compile(r"[.\w]+")

VHL_KEYWORDS = ("let", "print", "repeat", "input()")


def kind_of(path: str) -> str | None:
    if path.endswith(".vasm"):
        return "vasm"
    if path.endswith(".vhl"):
        return "vhl"
    return None


def diagnostics_for(text: str, kind: str) -> list[dict]:
    """[{line (0-based), message, severity: 'error'|'warning'}]"""
    out = []
    try:
        if kind == "vasm":
            prog = assemble(text).build()
        else:
            prog = compile_vhl(text).build()
    except LoadError as e:
        m = _LINE_RE.match(str(e))
        line = int(m.group(1)) - 1 if m else 0
        msg = m.group(2) if m else str(e)
        return [{"line": max(line, 0), "message": msg, "severity": "error"}]
    for w in prog.warnings:
        out.append({"line": 0, "message": w, "severity": "warning"})
    return out


def word_at(line_text: str, col: int) -> str | None:
    for m in _WORD_RE.finditer(line_text):
        if m.start() <= col <= m.end():
            return m.group(0)
    return None


def hover_for(word: str) -> str | None:
    entry = OP_DOCS.get(word.upper()) or DIRECTIVE_DOCS.get(word.lower())
    if entry is None:
        return None
    signature, effect = entry
    return f"**{signature}**\n\n{effect}"


def definition_line(text: str, word: str) -> int | None:
    """0-based line of `word:` (label) or `.chain word`."""
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        m = _LABEL_DEF_RE.match(line)
        if m and m.group(1) == word:
            return i
        m = _CHAIN_DEF_RE.match(line)
        if m and m.group(1) == word:
            return i
    return None


def completions_for(text: str, kind: str) -> list[tuple[str, str]]:
    """[(label, kind-tag)] — mnemonics, registers, guards, labels, chains."""
    if kind == "vhl":
        return [(k, "keyword") for k in VHL_KEYWORDS]
    items = [(mn, "opcode") for mn in sorted(MNEMONIC_TO_TRIPLE)]
    items += [(p, "opcode") for p in ("OUTC", "EXEC", "BR", "SEG", "SEGRAW", "OP")]
    items += [(d, "directive") for d in DIRECTIVE_DOCS]
    items += [(f"r{i}", "register") for i in range(4)]
    items += [(g, "guard") for g in sorted(GUARDS) if g.startswith(("+", "-"))]
    seen = set()
    for raw in text.splitlines():
        line = raw.strip()
        for regex, tag in ((_LABEL_DEF_RE, "label"), (_CHAIN_DEF_RE, "chain")):
            m = regex.match(line)
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                items.append((m.group(1), tag))
    return items


def main() -> int:  # pragma: no cover — protocol glue, exercised by editors
    try:
        from lsprotocol import types as lsp
        from pygls.server import LanguageServer
    except ImportError:
        import sys
        print("versor lsp needs the 'lsp' extra: pip install 'versor[lsp]'",
              file=sys.stderr)
        return 2

    server = LanguageServer("versor-lsp", "0.4.0")

    def publish(ls, uri: str):
        kind = kind_of(uri)
        if kind is None:
            return
        text = ls.workspace.get_text_document(uri).source
        sev = {"error": lsp.DiagnosticSeverity.Error,
               "warning": lsp.DiagnosticSeverity.Warning}
        diags = [
            lsp.Diagnostic(
                range=lsp.Range(lsp.Position(d["line"], 0),
                                lsp.Position(d["line"], 200)),
                message=d["message"], severity=sev[d["severity"]],
                source="versor")
            for d in diagnostics_for(text, kind)
        ]
        ls.publish_diagnostics(uri, diags)

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def did_open(ls, params):
        publish(ls, params.text_document.uri)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def did_change(ls, params):
        publish(ls, params.text_document.uri)

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def hover(ls, params):
        uri = params.text_document.uri
        doc = ls.workspace.get_text_document(uri)
        lines = doc.source.splitlines()
        if params.position.line >= len(lines):
            return None
        word = word_at(lines[params.position.line], params.position.character)
        if not word:
            return None
        text = hover_for(word)
        if text is None:
            return None
        return lsp.Hover(contents=lsp.MarkupContent(
            kind=lsp.MarkupKind.Markdown, value=text))

    @server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
    def definition(ls, params):
        uri = params.text_document.uri
        doc = ls.workspace.get_text_document(uri)
        lines = doc.source.splitlines()
        if params.position.line >= len(lines):
            return None
        word = word_at(lines[params.position.line], params.position.character)
        if not word:
            return None
        line = definition_line(doc.source, word)
        if line is None:
            return None
        return lsp.Location(uri=uri, range=lsp.Range(
            lsp.Position(line, 0), lsp.Position(line, len(word))))

    @server.feature(lsp.TEXT_DOCUMENT_COMPLETION)
    def completion(ls, params):
        uri = params.text_document.uri
        kind = kind_of(uri)
        if kind is None:
            return []
        doc = ls.workspace.get_text_document(uri)
        kinds = {"opcode": lsp.CompletionItemKind.Function,
                 "directive": lsp.CompletionItemKind.Keyword,
                 "register": lsp.CompletionItemKind.Variable,
                 "guard": lsp.CompletionItemKind.Constant,
                 "label": lsp.CompletionItemKind.Reference,
                 "chain": lsp.CompletionItemKind.Module,
                 "keyword": lsp.CompletionItemKind.Keyword}
        return [lsp.CompletionItem(label=label, kind=kinds[tag])
                for label, tag in completions_for(doc.source, kind)]

    server.start_io()
    return 0
