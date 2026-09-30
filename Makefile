# Tensorstead — make targets mapping to the four test tiers.

# PYTHONPATH="" isolates the uv-managed project venv from any host-level site-packages
# (a host-level tool can export one and shadow the project venv).
UV = PYTHONPATH= uv run

.PHONY: help require-operator-profile prepare-nodes build builds test test-unit test-contract test-integration test-hardware test-smoke lint typecheck coverage deadcode audit check ansible-check release-local deploy readiness hooks-install hooks-uninstall hooks-status hooks-install-server hooks-status-server services-status services-start services-restart services-stop services-boot-disable services-boot-enable reboot

# A local, ignored profile may safely name token files and the existing
# appliance endpoints. It must never contain a secret value.
-include .tensorstead/smoke.mk

# Local, ignored operator profile naming the inventory, settings, and Vault
# files for deploy/readiness. Like smoke.mk it holds paths only, never a
# secret value -- the Vault file it names stays encrypted and its password is
# still prompted for. Without this, every deploy meant retyping three absolute
# paths, which is a ritual rather than configuration.
-include .tensorstead/deploy.mk

# Host-changing targets deliberately require explicit local profile paths.  The
# artifact is computed from the checked-in release version so an old path in a
# long-lived operator settings file cannot accidentally redeploy an old wheel.
TENSORSTEAD_RELEASE_VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
TENSORSTEAD_DEPLOY_ARTIFACT ?= $(CURDIR)/dist/tensorstead-$(TENSORSTEAD_RELEASE_VERSION)-py3-none-any.whl
TENSORSTEAD_INVENTORY ?=
TENSORSTEAD_SETTINGS ?=
TENSORSTEAD_VAULT ?=

# How deploy/readiness authenticate. **Prompting is the default**, so a fresh
# clone with no local profile cannot be made non-interactive by accident, and
# nothing in this repository ever names a credential.
#
# An operator profile (.tensorstead/deploy.mk, git-ignored) may replace either with
# a non-interactive equivalent -- a vault password *script* that reads from the
# OS keychain, and an empty become flag once the target hosts permit passwordless
# sudo for the deploying user. Both are local decisions about local machines,
# which is why they live in the profile and not here.
TENSORSTEAD_VAULT_AUTH ?= --ask-vault-pass
TENSORSTEAD_BECOME_AUTH ?= --ask-become-pass

help:
	@echo "Safe local targets: build builds test test-unit test-contract test-integration lint typecheck check ansible-check release-local"
	@echo "Configured appliance target (read-only): test-smoke"
	@echo "Hardware target (explicit opt-in): test-hardware"
	@echo "Host-changing targets (explicit paths required): prepare-nodes deploy readiness"
	@echo "Service control (read-only): services-status"
	@echo "Service control (changes hosts): services-start services-restart services-stop"
	@echo "Boot arrangement only (starts nothing): services-boot-disable services-boot-enable"
	@echo "Reboot one node (names a host): reboot TENSORSTEAD_REBOOT_LIMIT=<host>"

# Build a wheel and append a local provenance record with an incrementing build
# number and SHA-256. This target never contacts an appliance.
build:
	$(UV) python scripts/build_package.py

# List the archived per-build wheels under dist/builds/ (build number + the git
# revision each was built from) so the build to roll back to is one glance, not
# a read of the JSONL ledger. See docs/builds.md#rollback.
builds:
	@for d in dist/builds/*/; do \
	  [ -d "$$d" ] || continue; \
	  m="$$(ls "$$d"*.build.json 2>/dev/null | head -1)"; \
	  rev=$$(python -c "import json,sys; print(json.load(open('$$m')).get('git_revision','?'))" 2>/dev/null || echo '?'); \
	  echo "$${d##*/builds/}  $${rev}"; \
	done

# Run every non-hardware test tier. Real hardware tests remain opt-in.
test:
	$(UV) pytest -m "not hardware and not hardware_disruptive" -q

test-unit:
	$(UV) pytest -m unit tests/unit -q

test-contract:
	$(UV) pytest -m contract tests/contract -q

test-integration:
	$(UV) pytest -m integration tests/integration -q

test-hardware:
	$(UV) pytest -m "hardware and not hardware_disruptive" tests/hardware --hardware -q

# Read only: checks API, MCP, and direct inference for an existing deployment.
# It fails before making a request until .tensorstead/smoke.mk names local token
# files and endpoints. It never creates, stops, restarts, or removes anything.
test-smoke:
	@test -n "$(TENSORSTEAD_SMOKE_API)" || (echo "Configure .tensorstead/smoke.mk from tests/smoke/profile.mk.example"; exit 2)
	$(UV) python scripts/smoke_test.py

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

# Coverage with a floor (see [tool.coverage.report] fail_under). The floor is
# not a target: an audit found paths no test had ever executed, including a
# certificate function called on every production request. Ratchet it up.
coverage:
	$(UV) pytest -m "not hardware and not hardware_disruptive" -q \
		--cov --cov-report=term-missing:skip-covered

# Static dead-code detection. contracts/ and domain/ are excluded: they are
# declarations -- Pydantic fields and enum members look unreferenced to any
# static tool, and naming each one would grow an ignore list nobody reads.
# ports/ likewise: a Protocol is satisfied structurally and never named.
#
# The dead-code items are resolved: mark_running is wired,
# and the duplicate node_resources and the unused name lookups were removed.
# The standing non-zero reports this target emits are pydantic
# @field_validator methods reached only through the validator hook, methods
# on the ratified test fakes, and small test helpers -- noise vulture cannot
# resolve statically. A non-zero exit is this target working, not broken.
#
# check_revision was ignored here *with a task attached* while the optimistic
# check sat unwired. It is gone: the check is now called by the PATCH
# route, and the dead second copy in lifecycle.py was deleted.
#
# A non-zero exit means vulture found something. That is the target working,
# not the target broken: at the time of writing it correctly reports
# FakePeerAgent as unused -- one of the six ratified test fakes
# that was built and never wired to a test.
deadcode:
	$(UV) vulture src/tensorstead tests scripts \
		--min-confidence 60 \
		--exclude "*/contracts/*,*/domain/*,*/ports/*" \
		--ignore-names "model_config,row_factory,build_agent_app_from_env,ContainerEngine,ServiceManager,_report_contract_version,noun,verb,purpose,requirements,pytestmark,read_count,pull_count,retained,retained_what,removed_what,running_image_digest,endpoint_authenticated,unexpected_instance,pytest_addoption,pytest_collection_modifyitems,_isolate_operator_config" \
		--ignore-decorators "@router.*,@app.*,@coordinator_app.*,@deployments_app.*,@nodes_app.*,@models_app.*,@runtimes_app.*,@images_app.*,@operations_app.*,@credentials_app.*,@buildspecs_app.*,@inferencekeys_app.*,@server.*"

# Everything that looks for code which exists but does nothing.
audit: coverage deadcode

typecheck:
	$(UV) mypy src tests

# The standard local quality gate. It neither connects to nor changes a Spark.
check: lint typecheck test coverage

# The gate, enforced locally. GitHub Actions runs the same `make check` on every
# push and pull request; these hooks catch problems before they leave the machine:
# pre-commit rejects unformatted or unlinted *staged* Python (about two seconds),
# pre-push runs `make check` in full.
#
# They live in ops/githooks rather than .git/hooks because .git/hooks is not
# version controlled, so a hook kept there is invisible to review, absent from a
# fresh clone, and silently different between one machine and the next. This
# points core.hooksPath at the tracked copy, which makes the gate a reviewable
# artifact like everything else here.
hooks-install:
	@git config core.hooksPath ops/githooks
	@echo "core.hooksPath -> ops/githooks"
	@echo "bypass a single command with TENSORSTEAD_SKIP_HOOKS=1 (or --no-verify)"

hooks-uninstall:
	@git config --unset core.hooksPath || true
	@echo "core.hooksPath cleared; hooks are no longer enforced"

# Tier 2: the gate on the server, which --no-verify cannot reach. See
# ops/githooks/server/README.md before running this -- it changes a box that
# serves this repository's git.
TENSORSTEAD_GIT_HOST ?= git.internal
TENSORSTEAD_GIT_DIR  ?= /srv/git/tensorstead.git
TENSORSTEAD_RUFF_VERSION := $(shell awk '/^name = "ruff"/{f=1} f&&/^version = /{gsub(/[",]/,"",$$3); print $$3; exit}' uv.lock)

hooks-install-server:
	@echo "installing ruff $(TENSORSTEAD_RUFF_VERSION) on $(TENSORSTEAD_GIT_HOST)"
	@ssh $(TENSORSTEAD_GIT_HOST) 'set -e; \
	  command -v ruff >/dev/null 2>&1 || python3 -m pip install --user --quiet "ruff==$(TENSORSTEAD_RUFF_VERSION)"; \
	  src=$$(command -v ruff || echo $$HOME/.local/bin/ruff); \
	  sudo install -m 0755 "$$src" /usr/local/bin/ruff; \
	  /usr/local/bin/ruff --version'
	@scp -q ops/githooks/server/pre-receive $(TENSORSTEAD_GIT_HOST):$(TENSORSTEAD_GIT_DIR)/hooks/pre-receive
	@ssh $(TENSORSTEAD_GIT_HOST) 'chmod +x $(TENSORSTEAD_GIT_DIR)/hooks/pre-receive'
	@echo "pre-receive installed at $(TENSORSTEAD_GIT_HOST):$(TENSORSTEAD_GIT_DIR)/hooks/"
	@echo "lockout recovery: ssh $(TENSORSTEAD_GIT_HOST) 'rm $(TENSORSTEAD_GIT_DIR)/hooks/pre-receive'"

hooks-status-server:
	@ssh $(TENSORSTEAD_GIT_HOST) 'echo "ruff: $$(/usr/local/bin/ruff --version 2>/dev/null || echo ABSENT)"; \
	  if [ -x $(TENSORSTEAD_GIT_DIR)/hooks/pre-receive ]; then echo "pre-receive: installed"; \
	  else echo "pre-receive: NOT installed"; fi'
	@echo "project pins ruff $(TENSORSTEAD_RUFF_VERSION) (from uv.lock)"

hooks-status:
	@path=$$(git config core.hooksPath || echo '<unset -- hooks NOT enforced>'); \
	echo "core.hooksPath: $$path"; \
	for h in pre-commit pre-push; do \
	  if [ -x ops/githooks/$$h ]; then echo "  $$h: present, executable"; \
	  elif [ -f ops/githooks/$$h ]; then echo "  $$h: present but NOT executable"; \
	  else echo "  $$h: MISSING"; fi; \
	done

# Validate Ansible content locally only. Syntax checks do not contact hosts.
ansible-check:
	$(UV) ansible-playbook -i ansible/inventories/example/hosts.yml ansible/playbooks/site.yml --syntax-check
	$(UV) ansible-playbook -i ansible/inventories/example/hosts.yml ansible/playbooks/readiness.yml --syntax-check
	$(UV) ansible-playbook -i ansible/inventories/example/hosts.yml ansible/playbooks/prepare-node.yml --syntax-check
	$(UV) ansible-playbook -i ansible/inventories/example/hosts.yml ansible/playbooks/services.yml --syntax-check
	$(UV) ansible-playbook -i ansible/inventories/example/hosts.yml ansible/playbooks/reboot.yml --syntax-check
	$(UV) ansible-lint ansible/

# Complete controller-only release gate: all checks plus a traceable wheel.
release-local: check ansible-check build

# Every host-changing target needs the same three paths. Saying which one is
# missing is not enough on its own: the useful half of the message is the
# command that creates the profile.
require-operator-profile:
	@missing=""; \
	test -n "$(TENSORSTEAD_INVENTORY)" || missing="$$missing TENSORSTEAD_INVENTORY"; \
	test -n "$(TENSORSTEAD_SETTINGS)"  || missing="$$missing TENSORSTEAD_SETTINGS"; \
	test -n "$(TENSORSTEAD_VAULT)"     || missing="$$missing TENSORSTEAD_VAULT"; \
	if [ -n "$$missing" ]; then \
		echo "Missing operator configuration:$$missing"; \
		echo ""; \
		if [ -f .tensorstead/deploy.mk ]; then \
			echo "  .tensorstead/deploy.mk exists but does not set all three."; \
			echo "  Compare it against ansible/deploy.mk.example."; \
		else \
			echo "  These live in a local, Git-ignored profile. Create it once:"; \
			echo ""; \
			echo "    mkdir -p .tensorstead"; \
			echo "    cp ansible/deploy.mk.example .tensorstead/deploy.mk"; \
			echo "    \$$EDITOR .tensorstead/deploy.mk"; \
			echo ""; \
			echo "  Then 'make deploy' and 'make readiness' need no arguments."; \
		fi; \
		echo ""; \
		echo "  See docs/operations.md#deploying-a-release."; \
		exit 2; \
	fi

# Install the current release wheel using protected operator configuration.
# This is intentionally separate from release-local: it contacts and changes
# hosts, and prompts for the Vault and sudo credentials rather than storing
# either in the repository or Make configuration.
deploy:
	@$(MAKE) --no-print-directory require-operator-profile
	@test -f "$(TENSORSTEAD_DEPLOY_ARTIFACT)" || (echo "Wheel not found: $(TENSORSTEAD_DEPLOY_ARTIFACT). Run make release-local first."; exit 2)
	$(UV) ansible-playbook -i "$(TENSORSTEAD_INVENTORY)" ansible/playbooks/site.yml \
		-e @"$(TENSORSTEAD_SETTINGS)" \
		-e @"$(TENSORSTEAD_VAULT)" \
		-e tensorstead_package_artifact="$(TENSORSTEAD_DEPLOY_ARTIFACT)" \
		$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)

# Bring freshly reimaged nodes to the state site.yml expects: access, headless
# boot, a reachable GRUB menu, and the ConnectX-7 fabric. Independent of deploy
# on purpose -- it is what you need when Tensorstead is *not* installed.
prepare-nodes:
	@$(MAKE) --no-print-directory require-operator-profile
	$(UV) ansible-playbook -i "$(TENSORSTEAD_INVENTORY)" ansible/playbooks/prepare-node.yml \
		-e @"$(TENSORSTEAD_SETTINGS)" \
		$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)

# Read-only counterpart to deploy.  It checks services, TLS, and coordinator
# reachability but never creates, stops, or changes a workload.
readiness:
	@$(MAKE) --no-print-directory require-operator-profile
	$(UV) ansible-playbook -i "$(TENSORSTEAD_INVENTORY)" ansible/playbooks/readiness.yml \
		-e @"$(TENSORSTEAD_SETTINGS)" \
		-e @"$(TENSORSTEAD_VAULT)" \
		$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)

# Start, restart, and inspect the three control-plane services without
# reinstalling anything.
#
# Starting is separate from arranging to start at boot. `services-start` brings
# them up now and changes no boot arrangement; `services-boot-disable` and
# `services-boot-enable` change only the boot arrangement and restart nothing.
# Conflating them is how an operator who wanted the services up right now gets a
# standing arrangement nobody asked for.
#
# `deploy` already enables and starts them, so this is not the only way -- it is
# the cheap way. `deploy` copies a wheel, reinstalls the venv, regenerates
# config and TLS, and refuses outright when the artifact for the checked-in
# release version is absent; none of which is what you want when a node came
# back from a reboot and you only need to know whether its services did too.
#
# None of these touch inference. Deployments run as their own Docker containers
# under their own units, so restarting the agent leaves a serving model serving
# -- the property that let build 125 deploy underneath a live TP=2 group.
TENSORSTEAD_SERVICE_PLAYBOOK = ansible/playbooks/services.yml

services-status:
	@$(MAKE) --no-print-directory require-operator-profile
	$(UV) ansible-playbook -i "$(TENSORSTEAD_INVENTORY)" $(TENSORSTEAD_SERVICE_PLAYBOOK) \
		-e @"$(TENSORSTEAD_SETTINGS)" \
		-e @"$(TENSORSTEAD_VAULT)" \
		-e tensorstead_service_state=status \
		$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)

services-start:
	@$(MAKE) --no-print-directory require-operator-profile
	$(UV) ansible-playbook -i "$(TENSORSTEAD_INVENTORY)" $(TENSORSTEAD_SERVICE_PLAYBOOK) \
		-e @"$(TENSORSTEAD_SETTINGS)" \
		-e @"$(TENSORSTEAD_VAULT)" \
		-e tensorstead_service_state=started \
		$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)

services-restart:
	@$(MAKE) --no-print-directory require-operator-profile
	$(UV) ansible-playbook -i "$(TENSORSTEAD_INVENTORY)" $(TENSORSTEAD_SERVICE_PLAYBOOK) \
		-e @"$(TENSORSTEAD_SETTINGS)" \
		-e @"$(TENSORSTEAD_VAULT)" \
		-e tensorstead_service_state=restarted \
		$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)

services-stop:
	@$(MAKE) --no-print-directory require-operator-profile
	$(UV) ansible-playbook -i "$(TENSORSTEAD_INVENTORY)" $(TENSORSTEAD_SERVICE_PLAYBOOK) \
		-e @"$(TENSORSTEAD_SETTINGS)" \
		-e @"$(TENSORSTEAD_VAULT)" \
		-e tensorstead_service_state=stopped \
		$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)

# Boot arrangement only. Neither of these starts, stops, or restarts anything.
#
# `deploy` converges to `tensorstead_services_enabled_at_boot`, which defaults to
# false -- so a deploy will not quietly re-enable what you disabled here. Set
# that variable true in your settings file if you want the opposite.
services-boot-disable:
	@$(MAKE) --no-print-directory require-operator-profile
	$(UV) ansible-playbook -i "$(TENSORSTEAD_INVENTORY)" $(TENSORSTEAD_SERVICE_PLAYBOOK) \
		-e @"$(TENSORSTEAD_SETTINGS)" \
		-e @"$(TENSORSTEAD_VAULT)" \
		-e tensorstead_service_boot=disabled \
		$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)

services-boot-enable:
	@$(MAKE) --no-print-directory require-operator-profile
	$(UV) ansible-playbook -i "$(TENSORSTEAD_INVENTORY)" $(TENSORSTEAD_SERVICE_PLAYBOOK) \
		-e @"$(TENSORSTEAD_SETTINGS)" \
		-e @"$(TENSORSTEAD_VAULT)" \
		-e tensorstead_service_boot=enabled \
		$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)

# Reboot ONE node. See ansible/playbooks/reboot.yml for why this is the
# ordinary reboot and not the emergency one: Ansible reboots over SSH, so it
# reaches only a node healthy enough not to need it. A node wedged the way a
# driver deadlock leaves it -- sshd unresponsive, text console
# still alive -- is the physical power button's problem, not this target's.
#
# The limit is required rather than defaulted. Deployments here span both Sparks
# as one group, so an unlimited reboot is an outage rather than a rolling
# restart, and the playbook refuses it too.
TENSORSTEAD_REBOOT_LIMIT ?=

reboot:
	@$(MAKE) --no-print-directory require-operator-profile
	@test -n "$(TENSORSTEAD_REBOOT_LIMIT)" || (\
		echo "Name the host to reboot, one at a time:"; \
		echo "  make reboot TENSORSTEAD_REBOOT_LIMIT=spark-beta.internal"; \
		echo "For a node that no longer answers SSH, this target cannot help --"; \
		echo "use the physical power button for a hard power cycle."; \
		exit 2)
	$(UV) ansible-playbook -i "$(TENSORSTEAD_INVENTORY)" ansible/playbooks/reboot.yml \
		--limit "$(TENSORSTEAD_REBOOT_LIMIT)" \
		-e @"$(TENSORSTEAD_SETTINGS)" \
		-e @"$(TENSORSTEAD_VAULT)" \
		$(TENSORSTEAD_VAULT_AUTH) $(TENSORSTEAD_BECOME_AUTH)
