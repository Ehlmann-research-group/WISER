# Environment setup


This page will teach us how to set up the prod environment and the dev environment for WISER
as well as what is going on behind the scenes.

You will need to have conda installed and python installed. You will also need to be able to use `make` in your terminal. Based on your operating system, you should look up how to install `make`. 

## Lockfiles

In WISER under /etc there are two very important files for environment setup: dev-conda-lock.yml
and prod-conda-lock.yml. These are our lockfiles for development and production respectively.
You should be very hesitant to change these files. You should also never directly edit them!

If you want to change the dependencies that these lockfiles have you will first have to decide
if the dependency should go into the prod lockfile or the dev lockfile. Dependencies in the prod
lockfile are needed for WISER to run in production. Dependencies in the dev lockfile are needed
to run WISER from source on github and do all of the tasks a developer would do like run the
tests or generate sphinx docs. Then you will have to follow the instructions below for each type of dependency you want to change.

### Changing production (prod) dependencies

To change dependencies for production you simply have to go into the file etc/wiser-prod.yml and
add the dependency. The dependency will either have to be a dependency on conda that we can get
from conda-forge or a dependency that we can get from pip. Add your dependency and version to the .yml file in the appropriate location (under dependencies if it's from conda and under pip if it's from PyPI).

It is encouraged to use as accurate of a version number as you can (so try to use major.minor.patch). We do not want our dependency versions to jump a lot when we add a new dependency. However, having big jumps in dependency versions may be unavoidable for important/large dependencies, like python, but we would still want to do this infrequently. This is important because plugin creators must partially base their plugin dependencies off WISER's dependencies.

### Changing development (dev) dependencies

To change dependencies for development you have to first go to the file etc/dev-additions.txt and add the dependency either under the _dependencies:_ line (for conda-forge dependencies) or under the _pip:_ line for pip dependencies. Follow the example at the top of the file. Note that you only use one equal sign to specify versions in this file. You would then go to the root directy and run the command:

`python src/devtools/make_dev_env.py etc/wiser-prod.yml etc/wiser-dev-additions.txt etc/wiser-dev.yml`

This command will generate the file wiser-dev.yml which will combine the packages in wiser-dev-additions  with the packages in wiser-prod.yml. You can also go into the folder _etc/_ and do `make dev-yaml`.

### Regenerating the lock file

To regenerate the dev lockfile you can do the command `conda-lock lock -f wiser-dev.yml` and then rename the lock file to _dev-conda-lock.yml_. To regenerate the prod lockfile you can do the command `conda-lock lock -f wiser-prod.yml` and rename the lockfile to _prod-conda-lock.yml_. 

You can also go into the folder _etc/_ and do `make create-lockfiles` to create both the dev and the prod lockfiles. To create just the dev do `make dev-lockfile` and to create just the prod do `make prod-lockfile`.

### Installing environment from lockfile

Then once you have regenerated the lock files, you can do `conda-lock install -n <conda-env-name> dev-conda-lock.yml`. Where conda-env-name is the name of the conda environment you want to install into. Additionally you can add the argument `--force-platform <OS-name>` to specify what OS you want the conda environment to support. This is useful when you are building an intel conda environment on an ARM mac as you can do the command: `conda-lock install -n wiser-intel --force-platform osx-64 dev-conda-lock.yml`. The text _osx-64_ means it will build to intel mac. 

If you are on MAC, to fully ensure your conda environment is the right platform, you may have to enter the command `conda config --env --set subdir osx-64` when you are in the conda enviroment you made. This specific command will ensure the environment is set to intel mac. To set to arm mac, you can do `conda config --env --set subdir osx-arm64`. You need only do these commands once.

You can also do the command `make install-dev-env` on windows to install the dev conda environment into conda. Do `make install-prod-env` to install the prod conda environment. If you are on MAC and you want to explicitly build for arm or intel you should do `make install-dev-env ENV=intel` to install for intel and `make install-dev-env ENV=arm` to install for arm.

### Listing out depenencies from lockfile

You may need to create a .yml file to show end users what dependencies WISER has in production. To do so do the following command `conda-lock render --kind env -p osx-64 -p osx-arm64 -p win-64 prod-conda-lock.yml`. This will print out the depenencies that wiser has for the specified platforms. Excluding all `-p <OS-version>` arguments will automatically create the .yml files for all the operaitng systems that _prod-conda-lock.yml_ supports.

**NOTE that the .yml file that `conda-lock render...` creates CAN NOT be substituted for the conda-lock.yml files!! Installing a conda environment based on the .yml files from the render command is not always reproducible. 
