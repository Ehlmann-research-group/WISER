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
	pyinstaller --name $(APP_NAME) --windowed --noconfirm \
		--osx-bundle-identifier $(OSX_BUNDLE_ID) \
		src/main.py

	# Patch the package with the correct version of libpng, since pyinstaller finds the wrong one.
	cp /opt/local/lib/libpng16.16.dylib dist/$(APP_NAME)/
	install_name_tool -id @loader_path/libpng16.16.dylib dist/$(APP_NAME)/libpng16.16.dylib
	install_name_tool -change /opt/local/lib/libz.1.dylib @loader_path/libz.1.dylib dist/$(APP_NAME)/libpng16.16.dylib
	cp dist/$(APP_NAME)/libpng16.16.dylib dist/$(APP_NAME).app/Contents/MacOS/

	# Generate a .dmg file containing the Mac application.
	hdiutil create dist/tmp.dmg -ov -volname "$(APP_NAME)" -fs HFS+ -srcfolder dist/$(APP_NAME).app
	hdiutil convert dist/tmp.dmg -format UDZO -o dist/$(APP_NAME)-$(APP_VERSION).dmg
	rm dist/tmp.dmg


clean:
	$(RM) -r build dist


.PHONY: lint typecheck build-mac clean
