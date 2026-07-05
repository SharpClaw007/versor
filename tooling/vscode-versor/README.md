# Versor for VS Code

Syntax highlighting for `.vasm` (Versor assembly) and `.vhl`, plus live
diagnostics, hover docs for every opcode, go-to-definition for labels and
chains, and completion — via the Versor language server.

## Setup

1. Install the language + server:
   `git clone https://github.com/SharpClaw007/versor && cd versor && pip install '.[lsp]'`
2. Ensure `versor` is on PATH (or set `versor.serverCommand` to e.g.
   `["/path/to/venv/bin/versor", "lsp"]`).
3. Install this extension from the `.vsix`:
   `code --install-extension versor-*.vsix`

## Building the .vsix

```bash
cd tooling/vscode-versor
npm install
npm run package
```
