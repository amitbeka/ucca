all: help

.PHONY: help clean tags test

help:
	@echo "Passible make targets:"
	@echo "make help -- this help message"
	@echo "make clean -- removes all file not tracked by git, except tag files"
	@echo "make tags -- (re-)creates tags and pycscope files"
	@echo "make test -- run tests directory"

clean:
	git clean -fxd -e cscope.out -e cscope.files -e tags

tags:
	rm -f cscope.* tags
	find `pwd` -name "*.py" > cscope.files
	ctags --python-kinds=-i -L cscope.files
	pycscope3 -R -i cscope.files

test:
	(cd ./tests && python3 -m unittest -v)
