Getting Started
===============

Download WISER
--------------

Pre-built installers for macOS and Windows are available at:
`ehlmann.caltech.edu/wiser <https://ehlmann.caltech.edu/wiser/index.html>`_

Download the installer for your platform and follow the on-screen instructions.
Linux users should use the AppImage distributed via
`GitHub Releases <https://github.com/Ehlmann-research-group/WISER/releases>`_.

.. note::

   The download location will change in a future release as WISER transitions
   to CU Boulder. Links on this page will be updated when that happens.

Running from Source
-------------------

If you want to run WISER from source or contribute to development, see the
:doc:`Developer Environment Setup <developer-content/environment-setup>` guide.

In brief:

.. code-block:: bash

   cd etc
   make install-dev-env          # macOS/Linux
   conda activate wiser-dev
   cd ../src
   python -m wiser

Supported Platforms
-------------------

WISER builds currently target:

- **macOS 15** --- ARM (Apple Silicon) and Intel
- **Windows 10/11**
- **Linux** --- Ubuntu 20.04+, Debian 11+, Fedora 39+ (amd64 and aarch64)

If you encounter issues building or running WISER, please
`open a GitHub Issue <https://github.com/Ehlmann-research-group/WISER/issues>`_.
