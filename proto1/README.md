# Prototype 1

This prototype app can load ENVI format image files and display them in a very
basic Qt GUI.

The main sophistication is in the ENVI loader, which is able to handle files
of any interleaving, endianness, and element type.  Files can be memory-mapped
to improve efficiency.

This prototype is able to load and visualize the 30GB test file within 10
seconds on Donnie's laptop.  For comparison, it takes about 1 minute to scan
the entire file's contents with `dd`.

Required packages:

*   numpy 1.17.0
*   PySide2 5.13.0 (Qt Python bindings)

To run:

Specify the ENVI data file path first, and the header path second.

```python gui.py envi-img-file envi-img-file.hdr```

