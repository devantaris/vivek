PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: setup test lint format data baseline sft reward eval help

help:
	@echo "PJ-03 targets:"
	@echo "  setup     - create .venv and install requirements"
	@echo "  test      - run pytest"
	@echo "  lint      - ruff check"
	@echo "  format    - ruff format + fix"
	@echo "  data      - download datasets (manual steps printed where needed)"
	@echo "  baseline  - Phase 1 baselines (stub until implemented)"
	@echo "  sft       - Phase 2 LoRA SFT (stub)"
	@echo "  reward    - Phase 3 GRPO/DPO (stub)"
	@echo "  eval      - Phase 4 evaluation (stub)"

setup:
	$(PYTHON) -m venv .venv
	.venv/Scripts/$(PYTHON) -m pip install --upgrade pip
	.venv/Scripts/$(PYTHON) -m pip install -r requirements.txt
	.venv/Scripts/$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/build_data.py

baseline:
	$(PYTHON) scripts/run_baseline.py

sft:
	$(PYTHON) scripts/run_sft.py

reward:
	$(PYTHON) scripts/run_reward.py

eval:
	$(PYTHON) scripts/run_eval.py
