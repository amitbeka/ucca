# first so calling make with no target will invoke the help message
all: help

.PHONY: help clean tags test intall dev-install full-install

# Path must end with a slash so installation will work
INSTALL_PATH = $(shell echo $${HOME}/lib/python/ucca/)
remove-install = $(shell rm -rf $(INSTALL_PATH))
create-install-dir = $(shell mkdir -p $(INSTALL_PATH))

help:
	@echo "Passible make targets:"
	@echo "make help -- this help message"
	@echo "make clean -- removes all file not tracked by git, except tag files"
	@echo "make dev-install -- install the library under the user's path using soft links"
	@echo "make full-install -- install the library under the user's path by copying"
	@echo "make tags -- (re-)creates tags and pycscope files"
	@echo "make test -- run tests directory"
	@echo "make uninstall - remove the installation directory"

clean:
	git clean -fxd -e cscope.out -e cscope.files -e tags

tags:
	rm -f cscope.* tags
	find `pwd` -name "*.py" > cscope.files
	ctags --python-kinds=-i -L cscope.files
	pycscope -R -i cscope.files

test:
	(cd ./tests && python3 -m unittest -v)

install:
	@echo "Use either full-install or dev-install as make targets:"
	@echo "dev-install will create ~/lib/python/ucca and soft link to these files."
	@echo "It is more suited for developers which want each change in the repository"
	@echo "to be reflected automatically in the run-time code."
	@echo "full-install will create ~/lib/python/ucca and copy the package files there."

dev-install:
	@echo Installing to $(INSTALL_PATH) ...
	$(call remove-install)
	$(call create-install-dir)
	$(shell for f in *.py; do ln -s `readlink -f $$f` $(INSTALL_PATH); done)

full-install:
	@echo Installing to $(INSTALL_PATH) ...
	$(call remove-install)
	$(call create-install-dir)
	$(shell for f in *.py; do cp `readlink -f $$f` $(INSTALL_PATH); done)

uninstall:
	@echo Removing from $(INSTALL_PATH) ...
	$(call remove-install)
