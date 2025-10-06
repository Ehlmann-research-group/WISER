# Environment setup


This page will teach us how to set up the prod environment and the dev environment for WISER
as well as what is going on behind the scenes.

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

To change dependencies for development you have to first go to the file etc/dev-additions.txt and add the dependency either under the `dependencies:` line (for conda-forge dependencies) or under the `pip:` line for pip dependencies. Follow the example at the top of the file. Note that you only use one equal sign to specify versions in this file. You would then go to the root directy and run the command:

`python src/devtools/make_dev_env.py etc/wiser-prod.yml etc/wiser-dev-additions.txt etc/wiser-dev.yml`

This command will generate the file wiser-dev.yml which will combine the packages in wiser-dev-additions  with the packages in wiser-prod.yml.

### Regenerating the lock file
...
- Talk about the lock environment-setup and what users will need to set them up
- Talk about the commands people can run to activate the lock environments 
- Talk about the script (and make the script) that people can run to refresh the lockfiles
    and refresh their environment (maybe make a make file that calls the scripts)
