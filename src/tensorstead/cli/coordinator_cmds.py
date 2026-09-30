"""Local coordinator commands — ``stead coordinator init|serve``.

Only these two commands run locally rather than as
HTTP calls, because they exist to create or run the API service itself. Every
other CLI command is an HTTP call to the coordinator.

- ``init`` creates the store (runs the SQLite migrations).
- ``serve`` runs the coordinator FastAPI application.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Annotated

import typer

from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate

coordinator_app = typer.Typer(
    name="coordinator",
    help="Local coordinator commands (create and run the API service).",
    no_args_is_help=True,
)

_DEFAULT_STORE = Path(os.environ.get("TENSORSTEAD_STORE", "./tensorstead.db"))
# src/tensorstead/adapters/sqlite/migrations
_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "adapters" / "sqlite" / "migrations"


@coordinator_app.command("init")
def init(
    store: Annotated[
        Path, typer.Option("--store", help="Path to the SQLite store.")
    ] = _DEFAULT_STORE,
) -> None:
    """Create the store and apply schema migrations."""
    parent = store.parent
    parent.mkdir(parents=True, exist_ok=True)
    conn = connect(store)
    applied = migrate(conn, _MIGRATIONS_DIR)
    conn.close()
    if applied:
        typer.echo(f"Store initialised at {store} — applied: {', '.join(applied)}")
    else:
        typer.echo(f"Store already initialised at {store} (no new migrations).")


def _is_loopback(host: str) -> bool:
    """Whether ``host`` only reaches this machine.

    Loopback is the whole ``127.0.0.0/8`` block for IPv4, not just
    ``127.0.0.1`` -- a hardcoded string list would refuse a deliberate bind
    to, say, ``127.0.0.2`` the same as it refuses a real network interface.
    ``"localhost"`` is a name, not an address, so it is checked literally.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@coordinator_app.command("serve")
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8080,
    store: Annotated[Path, typer.Option("--store")] = _DEFAULT_STORE,
    tls_cert: Annotated[
        Path | None,
        typer.Option("--tls-cert", envvar="TENSORSTEAD_TLS_CERT", help="Server certificate (PEM)."),
    ] = None,
    tls_key: Annotated[
        Path | None,
        typer.Option("--tls-key", envvar="TENSORSTEAD_TLS_KEY", help="Server private key (PEM)."),
    ] = None,
    insecure_dev_mode: Annotated[
        bool,
        typer.Option(
            "--insecure-dev-mode",
            help=(
                "Allow starting with no TENSORSTEAD_MGMT_TOKEN configured. Loopback binds "
                "only -- never overrides the off-loopback TLS/token requirement below."
            ),
        ),
    ] = False,
) -> None:
    """Run the coordinator API service (uvicorn).

    Deny by default: a management token is
    required unless ``--insecure-dev-mode`` says otherwise explicitly, and an
    off-loopback bind always requires both TLS and a token regardless of that
    flag -- there is no legitimate "insecure dev mode" for a listener the
    network can reach. This used to warn and continue on a missing token or
    a cleartext off-loopback bind; a fresh operator's first ``serve`` with no
    environment configured got a fully open management API and, at most, a
    line on stderr nothing was necessarily watching.

    Supplying ``--tls-cert`` and ``--tls-key`` serves the management API over
    HTTPS.
    """
    # Ensure the store exists before serving.
    if not store.exists():
        init(store=store)

    if (tls_cert is None) != (tls_key is None):
        raise typer.BadParameter("--tls-cert and --tls-key must be supplied together")

    loopback = _is_loopback(host)
    token_configured = bool(os.environ.get("TENSORSTEAD_MGMT_TOKEN"))

    if not loopback and tls_cert is None:
        raise typer.BadParameter(
            f"refusing to serve {host!r} without TLS: the management token would go "
            "on the wire in cleartext on every call. Supply --tls-cert/--tls-key, "
            "or bind to 127.0.0.1."
        )
    if not loopback and not token_configured:
        raise typer.BadParameter(
            f"refusing to serve {host!r} with no TENSORSTEAD_MGMT_TOKEN configured: "
            "an off-loopback listener with no token accepts every request from "
            "anything that can reach it. --insecure-dev-mode does not override this."
        )
    if loopback and not token_configured and not insecure_dev_mode:
        raise typer.BadParameter(
            "TENSORSTEAD_MGMT_TOKEN is not set. Set it, or pass --insecure-dev-mode "
            "to run an open coordinator on loopback deliberately."
        )

    # Imported here, not at module level: the CLI loads this module for every
    # command, and a client-only install has no web server to import.
    import uvicorn

    from tensorstead.coordinator.app import build_coordinator_app

    app = build_coordinator_app(store=store)
    if tls_cert is not None and tls_key is not None:
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_certfile=str(tls_cert),
            ssl_keyfile=str(tls_key),
        )
        return
    uvicorn.run(app, host=host, port=port)
