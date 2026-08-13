PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	pytest -q

reference:
	$(PYTHON) scripts/reproduce_reference_summary.py

run:
	$(PYTHON) scripts/run_paper_experiment.py --input data/nineteenFeaturesDf.csv --config configs/paper.yaml --output outputs/paper_run
