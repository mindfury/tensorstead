"""Host network facts the agent alone can establish.

A distributed runtime must be told which address to advertise to its peers and
which RDMA device carries its collectives. Neither is derivable from anything
the coordinator holds:

- ``Node.platform_facts`` is captured once at registration and deliberately
  never written back, so a node registered before a fact existed
  can never report it;
- the deployment record names an *interface*, which is stable and meaningful to
  an operator, while the address behind it is a runtime property of the host.

So the owning agent resolves both, at start, from the machine itself. Every
function here returns ``None`` when it cannot establish the fact rather than
guessing — the whole product exists because a value meaning "unknown" was once
rendered as a confident answer.
"""

from __future__ import annotations

import fcntl
import socket
import struct
from pathlib import Path

# Linux ioctl for "get the IPv4 address of this interface". The agent runs on
# DGX OS; this module is never exercised on the controller.
_SIOCGIFADDR = 0x8915

_SYS_CLASS_NET = Path("/sys/class/net")
_SYS_CLASS_INFINIBAND = Path("/sys/class/infiniband")


def interface_address(name: str) -> str | None:
    """The IPv4 address bound to ``name``, or ``None``.

    IPv4 specifically. The estate's node names resolve to public IPv6 and the
    collective backends fell back to loopback rather than use them.
    The address wanted here is the one on the RoCE link.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            packed = struct.pack("256s", name[:15].encode())
            return socket.inet_ntoa(fcntl.ioctl(probe.fileno(), _SIOCGIFADDR, packed)[20:24])
    except OSError:
        # No such interface, or it carries no IPv4. Both are "could not
        # establish", and both must reach the operator as that rather than as a
        # silently wrong address.
        return None


def _ipv4_gid_suffix(address: str) -> str:
    """``10.100.184.1`` -> ``ffff:0a64:b801``, the tail of its IPv4-mapped GID."""
    octets = [int(part) for part in address.split(".")]
    return f"ffff:{octets[0]:02x}{octets[1]:02x}:{octets[2]:02x}{octets[3]:02x}"


def rdma_gid_index(
    device: str,
    address: str,
    *,
    sys_class_infiniband: Path | None = None,
) -> int | None:
    """The RoCE v2 GID index for ``address`` on ``device``, or ``None``.

    RoCE offers several GIDs per port -- RoCE v1 and v2, IPv4-mapped and IPv6 --
    and the collective library must be told which. Left unset it picks one, and
    the wrong choice fails in ways that read as generic transport errors rather
    than as a misconfigured GID.

    **Why this is resolved here and not declared.** The recipe reads it from each
    node's own sysfs, and it differs per rank. Tensorstead carries one shared
    ``runtime_config`` across a deployment, so a declared literal would send one
    rank's index to both -- which a verification pass caught in the first draft of this work.
    This belongs with the other rank-local facts this module
    already resolves.

    Matched on two conditions rather than one: the GID must be RoCE **v2**, and
    it must encode this rank's own IPv4. Matching the type alone would pick the
    first v2 entry, which on a multi-address interface is not necessarily ours.
    """
    root = sys_class_infiniband or _SYS_CLASS_INFINIBAND
    suffix = _ipv4_gid_suffix(address)
    try:
        ports = sorted((root / device / "ports").iterdir())
    except OSError:
        return None

    for port in ports:
        try:
            entries = sorted(
                (port / "gids").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1
            )
        except OSError:
            continue
        for entry in entries:
            if not entry.name.isdigit():
                continue
            index = int(entry.name)
            try:
                gid = entry.read_text().strip().lower()
                kind = (port / "gid_attrs" / "types" / entry.name).read_text().strip()
            except OSError:
                continue
            if "v2" in kind.lower() and gid.endswith(suffix):
                return index
    return None


def rdma_device(name: str, *, sys_class_net: Path | None = None) -> str | None:
    """The RoCE device backing ``name`` (e.g. ``rocep1s0f0``), or ``None``.

    Read from sysfs rather than by running ``rdma`` or ``ibdev2netdev``: the
    mapping is a file-system fact, and shelling out to learn it would add a
    subprocess to the agent for something a directory listing answers.

    ``None`` for an ordinary Ethernet interface, which is a true answer — it has
    no RDMA device — and leaves the runtime to use its socket transport.
    """
    root = sys_class_net or _SYS_CLASS_NET
    try:
        devices = sorted(p.name for p in (root / name / "device" / "infiniband").iterdir())
    except OSError:
        return None
    return devices[0] if devices else None
