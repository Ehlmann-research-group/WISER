APP_VERSION := $(shell python ../../src/version.py)
MANUAL_TITLE:="WISER $(APP_VERSION) User Manual"

TARGETS=index.html contrast-stretch.html
PANDOC_OPTS=--standalone --css=style.css --metadata title=$(MANUAL_TITLE)


all: $(TARGETS)


%.html: %.md
	pandoc -f markdown_strict -t html $(PANDOC_OPTS) $< > $@

clean:
	$(RM) $(TARGETS)

.PHONY: all clean

