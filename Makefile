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

	./check_arch.sh


dist-mac : build-mac
	# Codesign the built application
	bash install-mac/sign_wiser.sh

	# Generate a .dmg file containing the Mac application.
	hdiutil create dist/tmp.dmg -ov -volname "$(APP_NAME)" -fs HFS+ \
		-srcfolder dist/$(APP_NAME).app
	hdiutil convert dist/tmp.dmg -format UDZO \
		-o dist/$(APP_NAME)-$(APP_VERSION).dmg
	rm dist/tmp.dmg

	# Submit the disk image to Apple for notarization
	xcrun notarytool submit dist/$(APP_NAME)-$(APP_VERSION).dmg \
		--apple-id $(AD_USERNAME) --team-id $(AD_TEAM_ID) --password $(AD_PASSWORD)

build-win : generated
	@set WISER_ENV=prod && pyinstaller WISER.spec

dist-win : build-win
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
	@python src/devtools/sign_mac.py --link "$(LINK)" --app-version "$(APP_VERSION)" \
			--apple-id "$(AD_USERNAME)" --team-id "$(AD_TEAM_ID)" \
			--app-password "$(AD_PASSWORD)" --app-name "$(APP_NAME)" \
			--artifact-name "$(MAC_DIST_GITHUB_NAME)"

sign-windows:  # Usage `make sign-windows LINK=https://github.com/Ehlmann-research-group/WISER/actions/runs/18478361575/artifacts/4259044563`
	@rem Fail if LINK is missing
	@if "$(LINK)"=="" ( echo ERROR: Provide LINK=<artifact URL> ; exit 1 )
	@rem Call Python script with args
	@python src\devtools\sign_windows.py --link "$(LINK)" --nsis $(NSIS) --app-version "$(APP_VERSION)" --sha1 "$(SHA1_THUMBPRINT)"

.PHONY: generated lint typecheck build-mac build-win clean sign-windows
