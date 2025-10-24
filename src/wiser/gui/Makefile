# Makefile to generate Python UI files from the Qt Designer .ui files.

UIC=pyside2-uic

# The *.ui files are stored in this subdirectory.
UI_SRC_DIR = ./ui_files

# The *_ui.py files are generated into this subdirectory.
TARGET_DIR = ./generated

FILENAMES=\
	about_dialog_ui.py \
	app_config_ui.py \
	band_chooser_ui.py \
	band_math_ui.py \
	bug_reporting_dialog_ui.py \
	channel_stretch_widget_ui.py \
	continuum_removal_dimensions_bands_ui.py \
	dataset_editor_ui.py \
	export_image_ui.py \
	export_plot_image_ui.py \
	export_spectra_dialog_ui.py \
	fits_dataset_dialog_ui.py \
	fits_spectra_dialog_ui.py \
	generic_spectral_computation_ui.py \
	geo_coords_dialog_ui.py \
	geo_referencer_dialog_ui.py \
	image_coords_widget_ui.py \
	import_spectra_text_ui.py \
	import_wavelengths_ui.py \
	interactive_scatter_plot_ui.py \
	reference_system_creator_ui.py \
	roi_info_editor_ui.py \
	save_dataset_advanced_details_ui.py \
	save_dataset_basic_details_ui.py \
	save_dataset_ui.py \
	scatter_plot_axes_ui.py \
	scatter_plot_colormap_ui.py \
	similarity_transform_dialog_ui.py \
	spectral_angle_mapper_ui.py \
	spectral_feature_fitting_ui.py \
	spectrum_info_editor_ui.py \
	spectrum_plot_config_ui.py \
	split_unsplit_pane_ui.py \
	stretch_config_widget_ui.py \
	subdataset_chooser_dialog_ui.py \
	system_info_ui.py

TARGETS=$(addprefix $(TARGET_DIR)/, $(FILENAMES))


all : generated

# Put a "generated" target here, so that we can use the same target name as the
# top-level Makefile target.
generated : $(TARGETS)

$(TARGET_DIR)/%_ui.py : $(UI_SRC_DIR)/%.ui
	$(UIC) $< -o $@

clean :
	$(RM) $(TARGETS)

.PHONY : all generated clean
