PYLINT=pylint
PYLINT_OPTS=

MYPY=mypy
MYPY_OPTS=

APP_NAME=ISWorkbench
APP_VERSION=0.0.1
OSX_BUNDLE_ID=edu.caltech.gps.ISWB

lint:
	find src -name "*.py" | xargs $(PYLINT) $(PYLINT_OPTS)

typecheck:
	$(MYPY) $(MYPY_OPTS) src

build-mac:
	# TODO:  Need to re-generate resources.py file for the platform!

	pyinstaller --name $(APP_NAME) --windowed --noconfirm \
		--osx-bundle-identifier $(OSX_BUNDLE_ID) \
		src/main.py

	# Patch the package with the correct version of libpng, since pyinstaller
	# finds the wrong one.  (Note:  Can't use the --add-binary option because
	# the dylib's information also needs to be patched.)
	cp /opt/local/lib/libpng16.16.dylib dist/$(APP_NAME)/
	install_name_tool -id @loader_path/libpng16.16.dylib dist/$(APP_NAME)/libpng16.16.dylib
	install_name_tool -change /opt/local/lib/libz.1.dylib @loader_path/libz.1.dylib dist/$(APP_NAME)/libpng16.16.dylib
	cp dist/$(APP_NAME)/libpng16.16.dylib dist/$(APP_NAME).app/Contents/MacOS/

	# Generate a .dmg file containing the Mac application.
	hdiutil create dist/tmp.dmg -ov -volname "$(APP_NAME)" -fs HFS+ -srcfolder dist/$(APP_NAME).app
	hdiutil convert dist/tmp.dmg -format UDZO -o dist/$(APP_NAME)-$(APP_VERSION).dmg
	rm dist/tmp.dmg


build-win:
	# TODO:  Need to re-generate resources.py file for the platform!

	# To debug PyInstaller issues:
	#   - drop the "--windowed" option
	#   - add a "--debug [bootloader|imports|noarchive|all]" option
	pyinstaller --name $(APP_NAME) --windowed --noconfirm \
		--exclude-module
		src/main.py

	# CAN'T GET --windowed TO WORK - FROZEN APP DOESN'T START :-(
	# Extra "--add-binary" arguments because PyInstaller won't properly
	# include all the necessary Qt DLLs.  The worst one is the SVG icon
	# resources, which require a few steps just to get the icons to even
	# show up in the frozen UI.
    pyinstaller --name $(APP_NAME) --noconfirm \
    	--add-binary C:\ProgramData\Miniconda3\Library\plugins\platforms;platforms \
    	--add-binary C:\ProgramData\Miniconda3\Library\plugins\iconengines;iconengines \
    	--hidden-import PySide2.QtSvg \
    	src\main.py

	# TODO:  Installer


clean:
	$(RM) -r build dist


.PHONY: lint typecheck build-mac build-win clean
