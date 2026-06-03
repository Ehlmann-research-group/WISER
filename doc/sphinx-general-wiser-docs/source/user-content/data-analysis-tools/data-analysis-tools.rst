Data Analysis Tools
===================

This section helps you understand how to use WISER's data analysis tools — the
algorithms for transforming, classifying, detecting, unmixing, and visualizing
hyperspectral imagery. Each page explains what the tool does, how it works, and
how to run it from the interface.

Transforms & Dimensionality Reduction
--------------------------------------

- :doc:`Principal Component Analysis (PCA) <pca>` — reduce a cube to a few bands
  ordered by variance.
- :doc:`Minimum Noise Fraction (MNF) <mnf>` — reduce a cube to bands ordered by
  signal-to-noise ratio.

Classification & Clustering
---------------------------

- :doc:`K-Means Clustering <kmeans>` — group pixels into clusters by spectral
  similarity.

Target Detection & Spectral Matching
------------------------------------

- :doc:`Mixture-Tuned Matched Filter (MTMF) <mtmf>` — detect a target material
  against an unknown background.
- :doc:`Spectral Angle Mapper (SAM) <spectral-angle-mapper>` — match pixels to
  reference spectra by spectral angle.
- :doc:`Spectral Feature Fitting (SFF) <spectral-feature-fitting>` — match
  pixels to reference spectra by their absorption features.

Unmixing
--------

- :doc:`Linear Unmixing <linear-unmixing>` — estimate per-pixel abundances of a
  set of endmember spectra.

Visualization
-------------

- :doc:`Interactive Scatter Plot <interactive-scatter-plot>` — explore two bands
  in feature space and link selections back to the image.

.. toctree::
   :hidden:

   pca
   mnf
   kmeans
   mtmf
   spectral-angle-mapper
   spectral-feature-fitting
   linear-unmixing
   interactive-scatter-plot
