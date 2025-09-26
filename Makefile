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
	MACOSX_DEPLOYMENT_TARGET=11.0 WISER_ENV=prod pyinstaller --clean --log-level=DEBUG --noconfirm WISER-macOS.spec > debug_output.txt 2>&1

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
	@set WISER_ENV=prod && pyinstaller WISER.spec > debug_output.txt 2>&1

dist-win : build-win
	$(NSIS) /NOCD /DWISER_VERSION="$(APP_VERSION)" /DSHA1_THUMBPRINT=$(SHA1_THUMBPRINT) install-win\win-install.nsi

clean:
	$(MAKE) -C src clean
	$(MAKE) -C src/wiser/gui clean

	$(RM) -r build dist


.PHONY: generated lint typecheck build-mac build-win clean
