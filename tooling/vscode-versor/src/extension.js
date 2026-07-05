const vscode = require("vscode");
const { LanguageClient } = require("vscode-languageclient/node");

let client;

function activate(context) {
  const command = vscode.workspace
    .getConfiguration("versor")
    .get("serverCommand", ["versor", "lsp"]);

  const serverOptions = {
    command: command[0],
    args: command.slice(1),
    options: {},
  };
  const clientOptions = {
    documentSelector: [
      { scheme: "file", language: "vasm" },
      { scheme: "file", language: "vhl" },
    ],
  };
  client = new LanguageClient("versor", "Versor Language Server",
    serverOptions, clientOptions);
  client.start().catch((e) => {
    vscode.window.showWarningMessage(
      `Versor LSP failed to start (${e.message}). Syntax highlighting still ` +
      "works; for diagnostics install the server: pip install 'versor[lsp]' " +
      "and ensure `versor` is on PATH (or set versor.serverCommand).",
    );
  });
  context.subscriptions.push({ dispose: () => client && client.stop() });
}

function deactivate() {
  return client ? client.stop() : undefined;
}

module.exports = { activate, deactivate };
