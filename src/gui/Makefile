# Makefile to generate Python UI files from the Qt Designer .ui files.

UIC=pyside2-uic

TARGETS=channel_stretch_widget_ui.py stretch_config_widget_ui.py

all: $(TARGETS)

%_ui.py : %.ui
	$(UIC) $< > $@

clean:
	$(RM) $(TARGETS)

.PHONY: all clean
