RCC=pyside2-rcc

RESOURCE_DIR=resources
TARGET_DIR=gui/generated

TARGETS=$(TARGET_DIR)/resources.py

PYTHON=python
TEST_MODULES=\
	tests.test_gui_util     \
	tests.test_gui_geom     \
	tests.test_raster_units


all : generated

# Put a "generated" target here, so that we can use the same target name as the
# top-level Makefile target.
generated : $(TARGETS)

$(TARGET_DIR)/%.py : $(RESOURCE_DIR)/%.qrc
	$(RCC) $< -o $@

test :
	$(PYTHON) -m unittest $(TEST_MODULES)

clean :
	rm $(TARGETS)

.PHONY : all generated clean
