# Overview
This release adds four new user-facing features: batch processing in BandMath, an interactive 2D scatter plot, a spectral feature fitting tool, and a spectral angle mapper tool. We also accelerated continuum removal on images by roughly 10× and fixed several EMIT-related bugs. Lastly, we implemented a solution for a plugin issue where some sub-packages used by plugins wouldn’t load. Details on the new features and the plugin fix are below.

# Upgrade Steps
- Download the new .dmg (macOS) or Windows installer and reinstall.

# Breaking Changes
- None known.

# New Features

## Bandmath Batch Processing
This feature lets you select an input folder of files to operate on from within the BandMath dialog. First, click **Enable Batch Processing**. Then choose the input folder you want to process. In the variables table, set the **Type** column to either **Batch – Image** or **Batch – Image Band**.

<img width="139" height="40" alt="image of type combo box batch options" src="https://github.com/user-attachments/assets/46460356-c398-4ff3-877a-d16f4d61f66a" />

When you assign a variable the type **Batch – Image** or **Batch – Image Band**, that variable is replaced with each file from the input folder in turn. For example, if your input folder contains `datasetB.tif` and `datasetC.tif`, and you set variable `a` in the expression `a + b` to **Batch – Image**, the expression will run twice: once with `a = datasetB.tif` and once with `a = datasetC.tif`.

Next, enter a **Suffix**, which is appended to the output filename for each new dataset. Finally, choose at least one output option—**Load Into WISER (in memory)** and/or **Output Folder (on disk)**—so WISER knows where to place the results.
<img width="287" height="84" alt="image" src="https://github.com/user-attachments/assets/66725489-a443-4bdb-ab71-07d5e53b4142" />

Click **Create Batch Job** to enqueue the job. A widget will appear in the right-hand table with details and controls to start, stop, cancel, or remove the job.

## Interactive Scatter Plot
You can open the interactive scatter plot from **Tools → Data Analysis → Interactive Scatter Plot**, or by right-clicking in the main image view and choosing **Data Analysis → Interactive Scatter Plot**.

<img width="389" height="165" alt="image" src="https://github.com/user-attachments/assets/9d7bac71-d895-436d-a1ea-fec41de063b2" />
<img width="461" height="113" alt="image" src="https://github.com/user-attachments/assets/194b9067-5ab8-4b82-94a6-fe019aa45fa5" />

After launching, select the dataset and band for the **X axis**, the dataset and band for the **Y axis**, and the dataset you want to **render highlights onto**. (The render target must have the same dimensions as the X/Y datasets.) Click **Create Plot**; a loading screen will appear. When ready, draw a polygon on the **scatter plot** to select points; the corresponding pixels will be highlighted on the chosen render dataset.
<img width="1480" height="527" alt="image" src="https://github.com/user-attachments/assets/f6286c5c-b1d2-46ad-97bd-e23cd8d2decb" />
You can create an ROI from the highlighted points by clicking **Create ROI from Selection**.

## Spectral Feature Fitting
This tool performs spectral feature fitting on a target spectrum. Load one or more reference spectra to compare against the target. Specify the **minimum** and **maximum** wavelengths for the feature window, and set an **RMSE Threshold** to filter the top matches. Collect a spectrum from your dataset to set it as the target.

You can find this tool in the same locations as the interactive scatter plot: **Tools → Data Analysis**, or the main raster view’s right-click **Data Analysis** submenu.

## Spectral Angle Mapper
This tool computes the spectral angle between two spectra over a specified wavelength range. The UI workflow is similar to spectral feature fitting; the primary difference is that you configure a **spectral angle threshold**.

You can find this tool in the same locations as the interactive scatter plot: **Tools → Data Analysis**, or the main raster view’s right-click **Data Analysis** submenu.

# Fixes

## Plugin Fix
Previously, when WISER was packaged with PyInstaller, it could include **package A** but omit unused sub-packages **B1**, **B2**, and **B3** to reduce app size. If a plugin depended on an omitted sub-package (e.g., **B1**), Python’s import system would still resolve to WISER’s bundled **package A**, which lacked **B1**, causing the plugin import to fail—even if the plugin attempted to ship or point to **B1** itself.

WISER now modifies part of the PyInstaller process to include all sub-packages of a package so these imports succeed. This change increases the app size, but avoids missing sub-packages at runtime. If you’re developing a plugin, it’s best practice to align your dependency versions with WISER’s to prevent unexpected behavior. You can find the package lists for each WISER build (Windows, macOS ARM, macOS Intel) here: <INSERT-HERE> (to be posted on the WISER-Plugin-API page).

# Known Issues
- JP2 file loading remains slow.
