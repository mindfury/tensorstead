"""Shared CLI constants and configuration resolution.

``cli/main.py`` and ``cli/commands.py`` both need the exit-code table and the
coordinator selector. Keeping them here (rather than importing from one CLI
module into the other) avoids the circular import between the Typer app
definition and its command groups, which mypy flags as ``has-type``.

**Where configuration comes from, and why it says so.** Each setting is
resolved from the first source that supplies it:

1. an explicit command-line flag (``--api``, ``--ca-bundle``), which the CLI
   applies to the environment before any command resolves settings,
2. an environment variable,
3. the config file (``~/.config/tensorstead/config.yml`` by default),
4. a built-in default.

Every resolution carries its origin, and ``stead status`` prints it. That
is not a nicety: an operator once found the CLI talking to the wrong address
with no way to tell whether the value came from their shell, a file, or a
default they had never seen. A tool that cannot say where its configuration
came from makes the user hunt for it, and hunting is the failure this
addresses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Exit-code table. `already_in_state` exits 0 deliberately: it is a
# satisfied request, not a failure.
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_UNREACHABLE = 3
EXIT_AUTH_REFUSED = 4

DEFAULT_API = "http://127.0.0.1:8080"

# Settings supplied by a command-line flag this invocation. The flag is applied
# to the environment so every command resolves through one path, and this set
# keeps the *reported source* honest: "from env" would be true of the mechanism
# and false to the operator, who typed a flag.
FLAG_SOURCED: set[str] = set()


def note_flag_source(env_var: str) -> None:
    """Record that ``env_var`` was set by a flag on this invocation."""
    FLAG_SOURCED.add(env_var)


@dataclass(frozen=True)
class Resolved:
    """One setting, and the source it came from.

    ``source`` is human-facing text such as ``"env TENSORSTEAD_API"`` or
    ``"config ~/.config/tensorstead/config.yml"``. It exists so a user never has
    to guess which of four places supplied a value.
    """

    value: str | None
    source: str

    def __bool__(self) -> bool:
        return bool(self.value)


def config_path() -> Path:
    """The config file this CLI reads, whether or not it exists.

    Honours ``TENSORSTEAD_CONFIG`` then ``XDG_CONFIG_HOME``, and otherwise sits
    beside the operator's other Tensorstead files in ``~/.config/tensorstead``.
    Reported by ``stead status`` even when absent, because "there is no
    config file, and here is where one would go" is the answer a new user
    needs.
    """
    explicit = os.environ.get("TENSORSTEAD_CONFIG", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "tensorstead" / "config.yml"


def load_config_file() -> dict[str, Any]:
    """Read the config file, or return an empty mapping.

    A malformed file returns empty rather than raising: the caller reports the
    resolved source, and every setting remains overridable by environment or
    flag, so a broken file must not make the CLI unusable.
    """
    path = config_path()
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    except yaml.YAMLError:
        # Caught by name, not via ValueError: YAMLError descends from
        # Exception, not from ValueError, so the obvious `except (OSError,
        # ValueError)` silently misses it and a typo in the config file
        # becomes a parser traceback in the user's face.
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _resolve(
    env_var: str, file_key: str, default: str | None = None, *, is_path: bool = False
) -> Resolved:
    value = os.environ.get(env_var, "").strip()
    if value:
        origin = "command-line flag" if env_var in FLAG_SOURCED else f"env {env_var}"
        return Resolved(value, origin)

    from_file = load_config_file().get(file_key)
    if isinstance(from_file, str) and from_file.strip():
        raw = from_file.strip()
        # Only filesystem settings get expanduser(): running a URL through
        # Path() collapses "//" into "/", which silently produced
        # "https:/host:8080" and an unexplainable connection refusal.
        resolved = str(Path(raw).expanduser()) if is_path else raw
        return Resolved(resolved, f"config {config_path()}")

    if default is not None:
        return Resolved(default, "built-in default")
    return Resolved(None, "not set")


def resolved_api_url() -> Resolved:
    """Coordinator base URL, with its origin."""
    return _resolve("TENSORSTEAD_API", "coordinator", DEFAULT_API)


def resolved_ca_bundle() -> Resolved:
    """Trust anchor for the coordinator's certificate, with its origin."""
    return _resolve("TENSORSTEAD_CA_BUNDLE", "ca_bundle", is_path=True)


def resolved_token() -> Resolved:
    """Management token, with its origin — never the value's contents.

    The config file names a *file* holding the token rather than the token
    itself, so an operator can show someone their configuration without
    redacting it (secrets stay out of ordinary configuration).
    """
    direct = os.environ.get("TENSORSTEAD_MGMT_TOKEN", "").strip()
    if direct:
        return Resolved(direct, "env TENSORSTEAD_MGMT_TOKEN")

    token_file = load_config_file().get("token_file")
    if isinstance(token_file, str) and token_file.strip():
        path = Path(token_file.strip()).expanduser()
        try:
            secret = path.read_text(encoding="utf-8").strip()
        except OSError:
            return Resolved(None, f"config token_file {path} (unreadable)")
        if secret:
            return Resolved(secret, f"config token_file {path}")
        return Resolved(None, f"config token_file {path} (empty)")

    return Resolved(None, "not set")


def api_url() -> str:
    """Return the coordinator base URL."""
    return resolved_api_url().value or DEFAULT_API


def ca_bundle() -> str | None:
    """Return the trust anchor for the coordinator, if one is configured.

    A private CA is not in any public trust store, so a remote client must be
    told about it explicitly.
    """
    return resolved_ca_bundle().value


def tls_verify() -> str | bool:
    """Return the ``verify`` argument for an httpx client.

    Returns the CA bundle path when one is configured and ``True`` otherwise,
    so the public trust store still applies. Verification is never disabled:
    a client that skips it would accept any certificate, which would leave the
    management token exactly as exposed as the plain HTTP this replaces.
    """
    return ca_bundle() or True
