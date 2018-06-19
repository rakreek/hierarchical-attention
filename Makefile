.PHONY: install clean-pyc test test-local test-all test-cover lint doc build

clean-pyc:
	find . -name '*.py[co]' -exec rm {} +

install: clean-pyc
	pip install --upgrade --no-deps ../fntk

test-all: clean-pyc
	nosetests --nologcapture --verbosity=3 --exclude-dir=tests/test_data 

test: clean-pyc
	nosetests --nologcapture --verbosity=3 --exclude-dir=tests/test_data --exclude-dir=tests/integration_tests

test-local: clean-pyc
	nosetests --nologcapture --verbosity=3 --exclude-dir=tests/test_data --exclude-dir=tests/integration_tests --exclude-dir=tests/server_tests

test-cover: clean-pyc
	nosetests --nologcapture --verbosity=3 --exclude-dir=tests/test_data  --exclude-dir=tests/server_tests --with-coverage --cover-html --cover-inclusive --cover-package=fntk

lint: clean-pyc
	-pylint fntk

doc: clean-pyc
	bash .build_docs.sh

build: lint test-cover doc
