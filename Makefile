.PHONY: build build-force

PYTHON ?= .venv/bin/python

# --- Build ---

build:
	cd frontend && npm run electron:build

build-force:
	cd frontend && npm run electron:build
