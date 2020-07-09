RCC=pyside2-rcc

RESOURCE_DIR=resources
TARGET_DIR=gui/generated

TARGETS=$(TARGET_DIR)/resources.py


all : generated

# Put a "generated" target here, so that we can use the same target name as the
# top-level Makefile target.
generated : $(TARGETS)

$(TARGET_DIR)/%.py : $(RESOURCE_DIR)/%.qrc
	$(RCC) $< -o $@

clean :
	rm $(TARGETS)

.PHONY : all generated clean
