# Makefile to generate Python UI files from the Qt Designer .ui files.

UIC=pyside2-uic

TARGETS=channel_stretch_widget_ui.py stretch_config_widget_ui.py \
	spectrum_plot_config_ui.py spectrum_info_editor_ui.py

all: $(TARGETS)

%_ui.py : %.ui
	$(UIC) $< > $@

clean:
	$(RM) $(TARGETS)

.PHONY: all clean
