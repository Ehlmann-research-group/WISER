APP_NAME=WISER
APP_VERSION := $(shell python src/wiser/version.py)

PYLINT=pylint
PYLINT_OPTS=

MYPY=mypy
MYPY_OPTS=

-include Secret.mk


#======================================================
# MACOSX SETTINGS

OSX_BUNDLE_ID=edu.caltech.gps.WISER

# Web-friendly arch token for the .dmg filename (x64 / arm64).
MAC_ARCH := $(shell uname -m | sed 's/x86_64/x64/')

#======================================================
# WINDOWS SETTINGS

NSIS="C:\Program Files (x86)\NSIS\makensis.exe"

MAC_DIST_GITHUB_NAME ?= wiser-macOS-X64

#======================================================
# BUILD RULES

all : generated


# Generate the various files that Qt will need for the UI.
generated :
	$(MAKE) -C src generated
	$(MAKE) -C src/wiser/gui generated


test:
	$(MAKE) -C src test

lint:
	find src -name "*.py" | xargs $(PYLINT) $(PYLINT_OPTS)

typecheck:
	$(MYPY) $(MYPY_OPTS) src


build-mac : generated
	@echo Building WISER version $(APP_VERSION)
	MACOSX_DEPLOYMENT_TARGET=11.0 WISER_ENV=prod pyinstaller --clean --log-level=DEBUG --noconfirm WISER-macOS.spec
	python src/devtools/patch_cv2_config_for_bundle.py dist/WISER/_internal/cv2
	./check_arch.sh


# Sign, notarize, staple, and package the locally built app. The same script runs in CI
# (.github/workflows/sign-macos.yml), so both paths produce identical artifacts.
# Both secret files are sourced into the environment rather than expanded into the recipe;
# sign_mac.py reads the credentials from there, so none of them reach process arguments.
dist-mac : build-mac
	@set -a; \
	[ -f ./Secret.sh ] && . ./Secret.sh; \
	[ -f ./Secret.mk ] && . ./Secret.mk; \
	set +a; \
	python src/devtools/sign_mac.py --app-path dist/$(APP_NAME).app \
		--app-name "$(APP_NAME)" --app-version "$(APP_VERSION)" --arch "$(MAC_ARCH)" --notarize

build-win : generated
	@set WISER_ENV=prod && pyinstaller WISER.spec
	python src/devtools/patch_cv2_config_for_bundle.py dist/WISER/_internal/cv2

# This should only be used for testing locally as it does not have some necessary edits to
# the final libraries that occurs in install-linux\multistage\Dockerfile and in
# install-linux\multistage_fedora\Dockerfile
build-linux : generated
	export WISER_ENV=prod
	pyinstaller WISER-ubuntu.spec
	python src/devtools/patch_cv2_config_for_bundle.py dist/WISER/_internal/cv2

dist-win : build-win
	$(NSIS) /NOCD /DWISER_VERSION="$(APP_VERSION)" /DSHA1_THUMBPRINT=$(SHA1_THUMBPRINT) install-win\win-install.nsi

quick-sign-win:
	$(NSIS) /NOCD /DWISER_VERSION="$(APP_VERSION)" /DSHA1_THUMBPRINT=$(SHA1_THUMBPRINT) install-win\win-install.nsi

# Note that these tests don't catch all issues that would occur on a new machine.
# To be more certain we catch problems, running with the github runner deployment
# pipeline is necessary
smoke-test-mac-build : build-mac
	./dist/WISER/WISER_Bin --test_mode

smoke-test-win-build : build-win
	./dist/WISER/WISER.exe --test_mode

clean:
	$(MAKE) -C src clean
	$(MAKE) -C src/wiser/gui clean

	$(RM) -r build dist

# Usage `make sign-mac LINK=https://github.com/Ehlmann-research-group/WISER/actions/runs/18481671108
# MAC_DIST_GITHUB_NAME=wiser-macOS-ARM64`
sign-mac:
	@if [ -z "$(LINK)" ]; then \
		echo "ERROR: Must provide LINK"; \
		exit 1; \
	fi

	@echo "Signing MacOS artifact from environment: $(ENV)"
	@echo "Downloading from: $(LINK)"
		@echo "Signing MacOS artifact from environment: $(ENV)"
	@echo "Downloading from: $(LINK)"
	@echo "App version: $(APP_VERSION)"
	@echo "Apple ID: $(AD_USERNAME)"
	@echo "Team ID: $(AD_TEAM_ID)"
	@echo "App Name: $(APP_NAME)"
	@set -a; \
	[ -f ./Secret.sh ] && . ./Secret.sh; \
	[ -f ./Secret.mk ] && . ./Secret.mk; \
	set +a; \
	python src/devtools/sign_mac.py --link "$(LINK)" --app-version "$(APP_VERSION)" \
			--app-name "$(APP_NAME)" \
			--artifact-name "$(MAC_DIST_GITHUB_NAME)" --notarize \
			$(if $(RELEASE_TAG),--release-tag "$(RELEASE_TAG)",)

sign-windows:  # Usage `make sign-windows LINK=https://github.com/Ehlmann-research-group/WISER/actions/runs/18478361575/artifacts/4259044563`
	@rem Fail if LINK is missing
	@if "$(LINK)"=="" ( echo ERROR: Provide LINK=<artifact URL> ; exit 1 )
	@rem Call Python script with args
	@python src\devtools\sign_windows.py --link "$(LINK)" --nsis $(NSIS) --app-version "$(APP_VERSION)" --sha1 "$(SHA1_THUMBPRINT)" $(if $(RELEASE_TAG),--release-tag "$(RELEASE_TAG)",)

# Mirror release assets to the Nexus raw repository. Normally this runs in CI
# (.github/workflows/backup-releases-to-nexus.yml); use this to backfill or to check the
# archive by hand. Credentials come from Secret.mk or the environment.
# Usage `make backup-releases TAG=v3.0b0`, or `make backup-releases ALL=1 DRY_RUN=1`.
backup-releases:
	@if [ -z "$(TAG)" ] && [ -z "$(ALL)" ]; then \
		echo "ERROR: Provide TAG=<tag> or ALL=1"; \
		exit 1; \
	fi
	@NEXUS_USERNAME="$(NEXUS_USERNAME)" NEXUS_PASSWORD="$(NEXUS_PASSWORD)" \
		python src/devtools/backup_releases_to_nexus.py \
		$(if $(TAG),--tag "$(TAG)",) $(if $(ALL),--all,) \
		$(if $(FORCE),--force,) $(if $(DRY_RUN),--dry-run,)

.PHONY: generated lint typecheck build-mac build-win clean sign-mac sign-windows backup-releases
