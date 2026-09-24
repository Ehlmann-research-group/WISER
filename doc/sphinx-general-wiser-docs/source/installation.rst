Installation
============

Download WISER
--------------

Pre-built installers for macOS, Windows and Linux are at
`lasp.colorado.edu/ehlmann-lab/wiser/ <https://lasp.colorado.edu/ehlmann-lab/wiser/>`_
and on `GitHub Releases <https://github.com/Ehlmann-research-group/WISER/releases>`_.

Download the installer for your platform and follow the on-screen instructions.

.. note::

   The download location will change in a future release as WISER transitions
   to CU Boulder. Links on this page will be updated when that happens.

Supported platforms
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Platform
     - Versions
   * - **macOS**
     - macOS 15 or newer, Apple Silicon (arm64) and Intel (x86_64)
   * - **Windows**
     - Windows 10 or 11, 64-bit
   * - **Linux**
     - Ubuntu 20.04+, Debian 11+, Fedora 39+ --- amd64 and aarch64

Hardware
--------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Requirement
   * - **CPU**
     - x86_64 (Intel/AMD) or arm64 (Apple Silicon / aarch64)
   * - **Memory**
     - 8 GB minimum; **16--32 GB recommended.** Band math and several analysis
       tools load their operands into memory, so a full flight line needs the
       larger figure
   * - **Storage**
     - ~1 GB for the installation; an SSD is strongly recommended, since WISER
       caches raster data to disk
   * - **GPU**
     - Not required

Running from source
-------------------

.. code-block:: bash

   git clone https://github.com/Ehlmann-research-group/WISER.git
   cd WISER/etc
   make install-dev-env          # macOS/Linux
   # On a Mac, name the architecture: make install-dev-env ENV=arm  (or ENV=intel)

   conda activate wiser-dev
   cd ../src
   python -m wiser

Full instructions, including Windows and the lockfile workflow, are in
:doc:`Environment Setup <developer-content/environment-setup>`.

Running from source is also how you get the data the
:doc:`tutorials <tutorials/index>` use --- the fixtures live in
``src/test_utils/`` and are not included in the installers.

Getting release announcements
-----------------------------

Send an email to ``sympa@lists.lasp.colorado.edu`` with the subject line
**subscribe wiser-announcements** and an empty body.

Trouble
-------

If WISER will not build or start, please
`open a GitHub Issue <https://github.com/Ehlmann-research-group/WISER/issues/new/choose>`_
with your platform, WISER version and the error text.
