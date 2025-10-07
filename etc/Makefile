# ---------- Makefile (portable) ---------------------------------------------

# Tools (override if needed, e.g., `make PY=py`)
PY ?= python

# Files
PROD_YML       := wiser-prod.yml
DEV_ADDITIONS  := wiser-dev-additions.txt
DEV_YML        := wiser-dev.yml

PROD_LOCK      := prod-conda-lock.yml
DEV_LOCK       := dev-conda-lock.yml

.PHONY: dev-yaml prod-lockfile dev-lockfile create-lockfiles \
        install-dev-env install-prod-env

## Build wiser-dev.yml from prod + additions
dev-yaml:
	$(PY) ../src/devtools/make_dev_env.py "$(PROD_YML)" "$(DEV_ADDITIONS)" "$(DEV_YML)"

## Create prod lockfile -> prod-conda-lock.yml
prod-lockfile: $(PROD_YML)
	conda-lock lock -f "$(PROD_YML)"
	$(PY) -c "import shutil; shutil.move('conda-lock.yml','$(PROD_LOCK)')"

## Create dev lockfile -> dev-conda-lock.yml
dev-lockfile: $(DEV_YML)
	conda-lock lock -f "$(DEV_YML)"
	$(PY) -c "import shutil; shutil.move('conda-lock.yml','$(DEV_LOCK)')"

## Run: dev-yaml -> prod-lockfile -> dev-lockfile
create-lockfiles: dev-yaml prod-lockfile dev-lockfile

# -------------------- install-dev-env ---------------------------------------
install-dev-env: $(DEV_LOCK)
ifeq ($(ENV),intel)
	@echo ENV=intel
	conda-lock install -n wiser-dev-intel --force-platform osx-64 "$(DEV_LOCK)"
	conda run -n wiser-dev-intel conda config --env --set subdir osx-64
else ifeq ($(ENV),arm)
	@echo ENV=arm
	conda-lock install -n wiser-dev-arm --force-platform osx-arm64 "$(DEV_LOCK)"
	conda run -n wiser-dev-arm conda config --env --set subdir osx-arm64
else ifeq ($(ENV),)
	@echo "ENV not set (default install)"
	conda-lock install -n wiser-dev "$(DEV_LOCK)"
else
	$(error Unknown ENV='$(ENV)'. Use ENV=intel or ENV=arm, or omit ENV)
endif

# -------------------- install-prod-env --------------------------------------
install-prod-env: $(PROD_LOCK)
ifeq ($(ENV),intel)
	@echo ENV=intel
	conda-lock install -n wiser-prod-intel --force-platform osx-64 "$(PROD_LOCK)"
	conda run -n wiser-prod-intel conda config --env --set subdir osx-64
else ifeq ($(ENV),arm)
	@echo ENV=arm
	conda-lock install -n wiser-prod-arm --force-platform osx-arm64 "$(PROD_LOCK)"
	conda run -n wiser-prod-arm conda config --env --set subdir osx-arm64
else ifeq ($(ENV),)
	@echo "ENV not set (default install)"
	conda-lock install -n wiser-prod "$(PROD_LOCK)"
else
	$(error Unknown ENV='$(ENV)'. Use ENV=intel or ENV=arm, or omit ENV)
endif