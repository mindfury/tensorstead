"""Typer CLI application.

The CLI is a *projection* of the coordinator API: it
holds no management logic of its own — every management command is an HTTP call
to the coordinator. ``TENSORSTEAD_API`` selects the coordinator.

Two commands are **not** management operations and run locally, because they
exist to create or run the API service itself:
``stead coordinator init`` and ``stead coordinator serve``.

Exit codes:
  0  success, including already_in_state
  1  operation failed with a reported reason
  2  invalid usage or validation rejection
  3  node or agent unreachable
  4  upstream authorization refused
"""

from __future__ import annotations

import os

import typer

from tensorstead.cli.commands import (
    buildspecs_app,
    credentials_app,
    deployments_app,
    images_app,
    inferencekeys_app,
    login_command,
    models_app,
    nodes_app,
    operations_app,
    runtimes_app,
    set_global_json_mode,
    status_command,
)
from tensorstead.cli.config import (
    EXIT_AUTH_REFUSED,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    EXIT_UNREACHABLE,
    EXIT_USAGE,
    note_flag_source,
)
from tensorstead.cli.coordinator_cmds import coordinator_app
from tensorstead.version import VERSION

# The exit-code table lives in cli/config.py (shared with the command groups to
# avoid a circular import); re-exported here for backward compatibility.
__all__ = [
    "EXIT_AUTH_REFUSED",
    "EXIT_FAILURE",
    "EXIT_SUCCESS",
    "EXIT_UNREACHABLE",
    "EXIT_USAGE",
]


app = typer.Typer(
    name="stead",
    help="Inference appliance management — deterministic, reproducible, inspectable.",
    no_args_is_help=True,
    add_completion=False,
)

# A global --json flag on every read command: a human reads state while
# agents need it parseable.
_json_flag = typer.Option(False, "--json", help="Emit JSON for machine consumption.")


def _show_version(value: bool) -> None:
    """Print the client version and exit.

    Answers "what am I running" without contacting a coordinator, which is the
    first question anyone asks and the one whose absence let a deploy report
    success while shipping the previous release.
    """
    if value:
        typer.echo(f"tensorstead {VERSION}")
        raise typer.Exit(EXIT_SUCCESS)


_version_flag = typer.Option(
    False,
    "--version",
    callback=_show_version,
    is_eager=True,
    help="Show the client version and exit.",
)


_api_flag = typer.Option(
    None, "--api", help="Coordinator base URL for this command, overriding config."
)
_ca_flag = typer.Option(None, "--ca-bundle", help="CA bundle for this command, overriding config.")


@app.callback()
def main(
    json: bool = _json_flag,
    version: bool = _version_flag,  # noqa: ARG001 - consumed by its eager callback
    api: str | None = _api_flag,
    ca_bundle: str | None = _ca_flag,
) -> None:
    """Tensorstead CLI."""
    set_global_json_mode(json)
    # Highest-precedence source, applied before any command resolves settings.
    # Exported rather than threaded through every command so `stead status`
    # reports it through the same path it reports every other source -- a
    # precedence order documented but not implemented would be worse than none.
    if api:
        os.environ["TENSORSTEAD_API"] = api
        note_flag_source("TENSORSTEAD_API")
    if ca_bundle:
        os.environ["TENSORSTEAD_CA_BUNDLE"] = ca_bundle
        note_flag_source("TENSORSTEAD_CA_BUNDLE")


@app.command("status")
def status(
    json_mode: bool = typer.Option(False, "--json", help="Emit JSON for machine consumption."),
) -> None:
    """Show what this client is connected to, and whether it is healthy."""
    status_command(json_mode)


@app.command("login")
def login(
    api: str = typer.Option(None, "--api", help="Coordinator base URL, e.g. https://host:8080."),
    ca_bundle: str = typer.Option(
        None, "--ca-bundle", help="Path to the managed private CA bundle."
    ),
    token_file: str = typer.Option(
        None, "--token-file", help="Path to a file holding the management token."
    ),
) -> None:
    """Save connection settings to a config file, after checking they work."""
    login_command(api, ca_bundle, token_file)


app.add_typer(coordinator_app)

# Phase 3 management command groups — thin HTTP clients of the API.
app.add_typer(nodes_app)
app.add_typer(models_app)
app.add_typer(runtimes_app)
app.add_typer(deployments_app)
# Phase 5 management command groups.
app.add_typer(images_app)
app.add_typer(buildspecs_app)
app.add_typer(inferencekeys_app)
app.add_typer(operations_app)
# Phase 7 management command group.
app.add_typer(credentials_app)


if __name__ == "__main__":
    app()
