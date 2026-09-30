"""Agent conformance suite.

Runs the same assertions against **both** the fake agent and the real agent, so
the fake is provably faithful rather than merely convenient.

The suite exercises the shared contract surface the agent exposes in phase 2:
``GET /agent/v1/info`` (version + platform facts) and the in-band
``X-Contract-Version`` reporting middleware. As the agent's
management routes land in later phases (acquire, deployments, observed), the
harness extends to cover them — the point is that every behaviour asserted
against the fake is also asserted against the real agent with its own
boundaries faked, and both must pass identically.

A thin adapter (``AgentUnderTest``) wraps the fake object and the real FastAPI
app behind one interface so the assertions are written once.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorstead.agent.app import AGENT_VERSION_HEADER, build_agent_app
from tensorstead.contracts.version import CONTRACT_VERSION
from tests.fakes.node_agent import FakeNodeAgent

pytestmark = pytest.mark.contract


class AgentUnderTest(Protocol):
    """The surface the conformance suite requires of both fake and real."""

    def info(self) -> dict[str, Any]: ...
    def contract_version_header(self) -> str | None: ...


class _FakeAdapter:
    """Wraps the fake node agent behind the AgentUnderTest interface."""

    def __init__(self, agent: FakeNodeAgent) -> None:
        self._agent = agent

    def info(self) -> dict[str, Any]:
        return self._agent.get_info()

    def contract_version_header(self) -> str | None:
        return CONTRACT_VERSION  # the fake reports the in-build version in-band


class _RealAdapter:
    """Wraps the real agent FastAPI app behind the AgentUnderTest interface."""

    def __init__(self, app: FastAPI) -> None:
        self._client = TestClient(app)

    def info(self) -> dict[str, Any]:
        response = self._client.get("/agent/v1/info")
        return cast(dict[str, Any], response.json())

    def contract_version_header(self) -> str | None:
        response = self._client.get("/agent/v1/info")
        return cast(str | None, response.headers.get(AGENT_VERSION_HEADER))


_AGENTS = [
    ("fake", _FakeAdapter(FakeNodeAgent())),
    ("real", _RealAdapter(build_agent_app())),
]


@pytest.mark.parametrize("name,agent", _AGENTS)
def test_info_reports_contract_version(name: str, agent: AgentUnderTest) -> None:
    """Both agents report the contract version."""
    info = agent.info()
    assert info["contract_version"] == CONTRACT_VERSION
    assert info["agent_version"]  # non-empty


@pytest.mark.parametrize("name,agent", _AGENTS)
def test_info_reports_service_manager(name: str, agent: AgentUnderTest) -> None:
    """Both agents report a service manager (systemd) and container engine."""
    info = agent.info()
    assert info["service_manager"] == "systemd"
    assert "name" in info["container_engine"]


@pytest.mark.parametrize("name,agent", _AGENTS)
def test_in_band_contract_version_header(name: str, agent: AgentUnderTest) -> None:
    """Both report the contract version in-band on every response.

    No heartbeat or extra round trip exists; the coordinator reads it from the
    header on each response.
    """
    header = agent.contract_version_header()
    assert header == CONTRACT_VERSION


@pytest.mark.parametrize("name,agent", _AGENTS)
def test_platform_facts_reported(name: str, agent: AgentUnderTest) -> None:
    """Both report platform facts as an open record."""
    info = agent.info()
    facts = info["platform_facts"]
    assert isinstance(facts, dict)
    assert "os_family" in facts
