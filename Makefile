APP_NAME=WISER
APP_VERSION=1.0a3

PYLINT=pylint
PYLINT_OPTS=

MYPY=mypy
MYPY_OPTS=


#======================================================
# MACOSX SETTINGS

OSX_BUNDLE_ID=edu.caltech.gps.WISER
OSX_KEY_NAME="WISER Project"

#======================================================
# WINDOWS SETTINGS

NSIS="C:\Program Files (x86)\NSIS\makensis.exe"


#======================================================
# BUILD RULES

all : generated


# Generate the various files that Qt will need for the UI.
generated :
	$(MAKE) -C src generated
	$(MAKE) -C src/gui generated


test:
	$(MAKE) -C src test

lint:
	find src -name "*.py" | xargs $(PYLINT) $(PYLINT_OPTS)

typecheck:
	$(MYPY) $(MYPY_OPTS) src


dist-mac : generated
	pyinstaller WISER-MacOSX.spec

	# Patch the package with the correct version of libpng, since pyinstaller
	# finds the wrong one.  (Note:  Can't use the --add-binary option because
	# the dylib's information also needs to be patched.)
	cp /opt/local/lib/libpng16.16.dylib dist/$(APP_NAME)/
	install_name_tool -id @loader_path/libpng16.16.dylib dist/$(APP_NAME)/libpng16.16.dylib
	install_name_tool -change /opt/local/lib/libz.1.dylib @loader_path/libz.1.dylib dist/$(APP_NAME)/libpng16.16.dylib
	cp dist/$(APP_NAME)/libpng16.16.dylib dist/$(APP_NAME).app/Contents/MacOS/

	# Codesign the application
	# codesign -s $(OSX_KEY_NAME) --deep dist/$(APP_NAME).app

	# Generate a .dmg file containing the Mac application.
	hdiutil create dist/tmp.dmg -ov -volname "$(APP_NAME)" -fs HFS+ -srcfolder dist/$(APP_NAME).app
	hdiutil convert dist/tmp.dmg -format UDZO -o dist/$(APP_NAME)-$(APP_VERSION).dmg
	rm dist/tmp.dmg


# To debug PyInstaller issues:
#   - drop the "--windowed" option
#   - add a "--debug [bootloader|imports|noarchive|all]" option
# TODO(donnie):  CAN'T GET --windowed TO WORK - FROZEN APP DOESN'T START :-(
# Extra "--add-binary" arguments because PyInstaller won't properly
# include all the necessary Qt DLLs.  The worst one is the SVG icon
# resources, which require a few steps just to get the icons to even
# show up in the frozen UI.
dist-win : generated
	pyinstaller --name $(APP_NAME) --noconfirm \
		--add-binary C:\ProgramData\Miniconda3\Library\plugins\platforms;platforms \
		--add-binary C:\ProgramData\Miniconda3\Library\plugins\iconengines;iconengines \
		--add-data C:\ProgramData\Miniconda3\Library\bin\libiomp5md.dll;. \
		--hidden-import PySide2.QtSvg \
		src\main.py

	$(NSIS) /NOCD install-win\win-install.nsi

clean:
	$(MAKE) -C src clean
	$(MAKE) -C src/gui clean

	$(RM) -r build dist


.PHONY: generated lint typecheck build-mac build-win clean
