CONFIG ?= configs/synthetic.yaml
ARTIFACTS ?= artifacts/synthetic

.PHONY: install lint test test-fast run backtest diagnose app docker clean

install:
	pip install -r requirements.lock
	pip install -e . --no-deps

lint:
	ruff check src tests app infra scripts

test:
	pytest

test-fast:
	pytest -m "not slow"

run:
	windcast run-all --config $(CONFIG)

backtest:
	windcast backtest --config $(CONFIG)

diagnose:
	python scripts/diagnose_data.py --config $(CONFIG)

app:
	streamlit run app/streamlit_app.py -- --artifacts $(ARTIFACTS)

docker:
	docker build -t windcast:latest .

clean:
	rm -rf data artifacts .pytest_cache .ruff_cache
