# Makefile to generate Python UI files from the Qt Designer .ui files.

UIC=pyside2-uic

# The .ui files are stored in this subdirectory.
UI_SRC_DIR = ./ui_files

TARGETS=about_dialog_ui.py \
	band_chooser_ui.py \
	channel_stretch_widget_ui.py \
	export_image_ui.py \
	spectrum_info_editor_ui.py \
	spectrum_plot_config_ui.py \
	stretch_config_widget_ui.py \
	system_info_ui.py

all: $(TARGETS)

%_ui.py : $(UI_SRC_DIR)/%.ui
	$(UIC) $< > $@

clean:
	$(RM) $(TARGETS)

.PHONY: all clean
