PYLINT=pylint
PYLINT_OPTS=

MYPY=mypy
MYPY_OPTS=

lint:
	find src -name "*.py" | xargs $(PYLINT) $(PYLINT_OPTS)

typecheck:
	$(MYPY) $(MYPY_OPTS) src


