.PHONY: help install dev test clean release

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help     Show this help message (default)"
	@echo "  install  Install the package"
	@echo "  dev      Install in editable/development mode"
	@echo "  test     Run the test suite"
	@echo "  clean    Remove build artifacts"
	@echo "  release  Build and upload to PyPI"

install:
	pip install .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info

release: clean
	python -m build
	twine upload dist/*
