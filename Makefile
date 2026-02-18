.PHONY: bench-smoke bench-full bench-gate bench-list build build-force install-hooks

PYTHON ?= .venv/bin/python

# --- Benchmark ---

bench-smoke:
	$(PYTHON) benchmarks/run.py --engine triaxis --queries smoke --tag "smoke_$$(date +%Y%m%d)"

bench-full:
	$(PYTHON) benchmarks/run.py --engine triaxis --queries full --tag "full_$$(date +%Y%m%d)"

bench-gate:
	$(PYTHON) benchmarks/run.py --gate --baseline latest --run-file $$(ls -t benchmarks/runs/*.json 2>/dev/null | head -1)

bench-list:
	$(PYTHON) benchmarks/run.py --list

# --- Build with gate ---

build: bench-smoke
	cd frontend && npm run electron:build

build-force:
	cd frontend && npm run electron:build

# --- Setup ---

install-hooks:
	bash scripts/install-hooks.sh
