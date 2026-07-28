# Code Documentation

This section walks through the internals of how major systems in WISER work.
Each page traces a single subsystem in depth — the classes involved, the
events and signals they exchange, and which component is responsible for what.
This should be used as a guide for developers reading, debugging, or extending that part of the
codebase. Please keep this up to date.

```{toctree}
:hidden:
:maxdepth: 1

viewport-system.md
spectrum-plot.md
rendering-pipeline.md
band-chooser.md
stretch-builder.md
data-caching.md
bandmath-internals.md
app-state.md
georeferencer-internals.md
crs-creator-internals.md
plugin-system.md
mosaic-internals.md
save-to-project.md
raster-format-dispatch.md
```
