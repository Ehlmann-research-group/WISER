RCC=pyside2-rcc

RESOURCE_DIR=wiser/resources
TARGET_DIR=wiser/gui/generated

TARGETS=$(TARGET_DIR)/resources.py

PYTHON=python
TEST_MODULES=\
	tests.test_gui_util           \
	tests.test_gui_geom           \
	tests.test_gui_spectrum_plot  \
	tests.test_raster_utils       \
	tests.test_bandmath_parser    \
	tests.test_bandmath_evaluator \
	tests.test_plugins_utils


all : generated

# Put a "generated" target here, so that we can use the same target name as the
# top-level Makefile target.
generated : $(TARGETS)

$(TARGET_DIR)/%.py : $(RESOURCE_DIR)/%.qrc
	$(RCC) $< -o $@

test :
	$(PYTHON) -m unittest $(TEST_MODULES)

clean :
	$(RM) $(TARGETS)

.PHONY : all generated clean
