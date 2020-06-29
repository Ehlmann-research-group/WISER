This directory contains `.ui` files created using the
[Qt Creator](https://doc.qt.io/qtcreator/index.html) tool, for the Imaging
Spectroscopy Workbench.

The `.ui` files are processed by the `pyside2-uic` tool to generate Python UI
code for inclusion into the application.  Since the corresponding Python code
is generated, it would normally not be included in the source code repository,
but it's difficult to generate the Python code on some platforms.  So, for the
time being, the generated Python sources are also checked into the repository.

The `Makefile` for generating the Python sources is in the parent directory
of this directoy.
