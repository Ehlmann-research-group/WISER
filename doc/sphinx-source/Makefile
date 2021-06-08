SOURCES=bandmath_plugins.rst \
	index.rst \
	plugins.rst \
	supporting_types.rst \
	ui_plugins.rst


all: runsphinx


runsphinx: $(SOURCES)
	sphinx-build . build

clean:
	$(RM) -r build


.PHONY: all runsphinx clean
