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
	pyinstaller --noconfirm WISER-macOS.spec


dist-mac : build-mac
	# Codesign the built application
	codesign -s "$(AD_CODESIGN_KEY_NAME)" --deep --force \
		--entitlements install-mac/entitlements.plist \
		-o runtime dist/$(APP_NAME).app

	# Generate a .dmg file containing the Mac application.
	hdiutil create dist/tmp.dmg -ov -volname "$(APP_NAME)" -fs HFS+ \
		-srcfolder dist/$(APP_NAME).app
	hdiutil convert dist/tmp.dmg -format UDZO \
		-o dist/$(APP_NAME)-$(APP_VERSION).dmg
	rm dist/tmp.dmg

	# Submit the disk image to Apple for notarization
	xcrun notarytool submit dist/$(APP_NAME)-$(APP_VERSION).dmg \
		--apple-id $(AD_USERNAME) --team-id $(AD_TEAM_ID) --password $(AD_PASSWORD)

# To debug PyInstaller issues:
#   - drop the "--windowed" option
#   - add a "--debug [bootloader|imports|noarchive|all]" option
# TODO(donnie):  CAN'T GET --windowed TO WORK - FROZEN APP DOESN'T START :-(
# Extra "--add-binary" arguments because PyInstaller won't properly
# include all the necessary Qt DLLs.  The worst one is the SVG icon
# resources, which require a few steps just to get the icons to even
# show up in the frozen UI.
dist-win : generated
	pyinstaller --log-level=DEBUG --name $(APP_NAME) --noconfirm \
	    --icon icons\wiser.ico \
		--add-binary C:\Users\jgarc\anaconda3\envs\wiser-source\Library\plugins\platforms;platforms \
		--add-binary C:\Users\jgarc\anaconda3\envs\wiser-source\Library\plugins\iconengines;iconengines \
		--add-binary C:\Users\jgarc\anaconda3\envs\wiser-source\Library\lib\gdalplugins\gdal_FITS.dll;gdalplugins \
		--add-binary C:\Users\jgarc\anaconda3\envs\wiser-source\Library\lib\gdalplugins\gdal_netCDF.dll;gdalplugins \
		--add-binary C:\Users\jgarc\anaconda3\envs\wiser-source\Library\lib\gdalplugins\gdal_HDF4.dll;gdalplugins \
		--add-binary C:\Users\jgarc\anaconda3\envs\wiser-source\Library\lib\gdalplugins\gdal_HDF5.dll;gdalplugins \
		--add-data src\wiser\bandmath\bandmath.lark;wiser\bandmath \
		--hidden-import PySide2.QtSvg --hidden-import PySide2.QtXml \
		--collect-all osgeo \
		--exclude-module PyQt5 \
		src\wiser\__main__.py > debug_output.txt 2>&1

	$(NSIS) /NOCD /DWISER_VERSION="$(APP_VERSION)" /DSHA1_THUMBPRINT=$(SHA1_THUMBPRINT) install-win\win-install.nsi

clean:
	$(MAKE) -C src clean
	$(MAKE) -C src/wiser/gui clean

	$(RM) -r build dist


.PHONY: generated lint typecheck build-mac build-win clean
