"""Static regression checks for explicit, artifact-safe Make deployment targets."""

from __future__ import annotations

from pathlib import Path


def test_deploy_requires_explicit_profiles_and_overrides_stale_artifact_paths() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "TENSORSTEAD_INVENTORY ?=" in makefile
    assert "TENSORSTEAD_SETTINGS ?=" in makefile
    assert "TENSORSTEAD_VAULT ?=" in makefile
    assert 'tensorstead_package_artifact="$(TENSORSTEAD_DEPLOY_ARTIFACT)"' in makefile
    assert "Run make release-local first." in makefile
    # The checked-in default is to prompt for both. Deploy authentication moved
    # behind two variables so a Git-ignored operator profile can supply
    # non-interactive equivalents -- a keychain-backed vault script, and an empty
    # become flag where the hosts permit passwordless sudo. What must stay true
    # is that *this file* asks: a fresh clone with no local profile cannot be
    # made non-interactive by accident, and nothing here names a credential.
    assert "TENSORSTEAD_VAULT_AUTH ?= --ask-vault-pass" in makefile
    assert "TENSORSTEAD_BECOME_AUTH ?= --ask-become-pass" in makefile
    assert "$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)" in makefile


def test_the_makefile_never_names_a_credential() -> None:
    """Deploy authentication is configurable; a secret in this file is not.

    The point of routing vault and become authentication through variables is
    that a *local, ignored* profile supplies them. If a password, a token, or a
    path masquerading as one ever lands in the committed Makefile, that
    convenience has become a credential in version control.
    """
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for forbidden in ("--vault-password-file", "BECOME_PASSWORD", "ANSIBLE_BECOME_PASS"):
        assert forbidden not in makefile, (
            f"{forbidden!r} appears in the committed Makefile; non-interactive "
            f"authentication belongs in .tensorstead/deploy.mk, which is Git-ignored"
        )


def test_readiness_target_is_separate_from_host_changing_deploy() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    readiness = makefile.split("readiness:\n", maxsplit=1)[1]
    assert "ansible/playbooks/readiness.yml" in readiness
    assert "ansible/playbooks/site.yml" not in readiness


def test_deploy_reads_a_local_operator_profile() -> None:
    """`make deploy` must not require three absolute paths on every invocation.

    `make test-smoke` already took its configuration from a Git-ignored
    `.tensorstead/smoke.mk`; deploy did not, so every deployment meant retyping
    three absolute paths. That is a ritual, not configuration, and it is the
    same friction the CLI's environment variables caused.
    """
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "-include .tensorstead/deploy.mk" in makefile
    assert Path("ansible/deploy.mk.example").is_file(), (
        "a template must exist, or the profile is undiscoverable"
    )


def test_the_missing_profile_message_names_the_command_that_fixes_it() -> None:
    """An error that states a precondition without the remedy is half done."""
    makefile = Path("Makefile").read_text(encoding="utf-8")
    check = makefile.split("require-operator-profile:\n", maxsplit=1)[1].split("\n\n")[0]

    assert "cp ansible/deploy.mk.example .tensorstead/deploy.mk" in check
    assert "Missing operator configuration" in check
    # Every host-changing target must use the shared check rather than drifting
    # into its own copy. Enumerated rather than counted: the count was 2, a new
    # host-changing target made it 3, and a number that has to be edited every
    # time a target is added tests the number rather than the property.
    for target in ("prepare-nodes", "deploy", "readiness"):
        recipe = makefile.split(f"\n{target}:\n", maxsplit=1)[1].split("\n\n")[0]
        assert "$(MAKE) --no-print-directory require-operator-profile" in recipe, (
            f"the {target!r} target changes a host without checking that the operator "
            f"profile is configured"
        )


def test_the_profile_template_carries_no_secret_value() -> None:
    """The Vault file is named, never opened; its password is still prompted for."""
    template = Path("ansible/deploy.mk.example").read_text(encoding="utf-8")

    assert "TENSORSTEAD_VAULT" in template
    assert "PATHS ONLY" in template.upper()
    for leak in ("password", "token", "secret ="):
        assert f"{leak}=" not in template.lower().replace(" ", "")
