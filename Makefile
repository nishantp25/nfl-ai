.PHONY: install verify ingest validate test lint clean doctor example

install:
	pip install -e ".[dev]"

verify:            ## probe the provider for real coverage
	nfldl verify

verify-deep:       ## also sample one season per dataset (slower, checks schemas + keys)
	nfldl verify --sample

sync-manifest:     ## write discovered coverage back into configs/data_sources.yaml
	nfldl verify --write-manifest

ingest:            ## fetch the required datasets for the target window
	nfldl ingest --required-only

ingest-all:
	nfldl ingest

validate:
	nfldl validate

doctor:
	nfldl doctor

example:
	python examples/pregame_frame.py

test:
	python -m pytest tests -q

lint:
	ruff check src tests examples
	mypy src --ignore-missing-imports

clean:             ## remove the lake and reports, keep configs
	rm -rf data/raw data/interim data/_cache reports

# A cold start, in order.
bootstrap: install verify ingest validate
