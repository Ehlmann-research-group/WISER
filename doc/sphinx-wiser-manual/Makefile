SOURCES=index.rst \
	bandmath.rst


all: runsphinx


runsphinx: $(SOURCES)
	sphinx-build . build

clean:
	$(RM) -r build


.PHONY: all runsphinx clean
