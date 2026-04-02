.PHONY: install dev test clean

install:
	pip install .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
