.DEFAULT_GOAL := help

PYTHON         ?= python
PORT           ?= 8001
HOST           ?= 127.0.0.1
SQLITE_PATH    ?= ./fitpace.db
SQLITE_URL     := sqlite+aiosqlite:///$(SQLITE_PATH)

export PYTHONPATH := .

.PHONY: help install data train db-init setup run test demo clean \
        docker-up docker-down docker-seed

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

data: ## Generate synthetic_progress.csv
	$(PYTHON) data/generate_synthetic.py

train: ## Train the regression model and write app/ml/model.pkl
	$(PYTHON) app/ml/train.py

db-init: ## Create the SQLite schema at $(SQLITE_PATH)
	@rm -f $(SQLITE_PATH)
	DATABASE_URL="$(SQLITE_URL)" $(PYTHON) scripts/init_db.py

setup: install data train db-init ## Install deps, generate data, train model, init DB

run: ## Run the API against SQLite on http://$(HOST):$(PORT)
	DATABASE_URL="$(SQLITE_URL)" $(PYTHON) -m uvicorn app.main:app \
		--host $(HOST) --port $(PORT) --reload

test: ## Run the pytest suite
	$(PYTHON) -m pytest tests/ -q

demo: ## Walk through every endpoint against a running API (defaults to http://$(HOST):$(PORT))
	FITPACE_URL="http://$(HOST):$(PORT)" $(PYTHON) scripts/demo.py

clean: ## Remove generated artifacts (db, model, csv)
	rm -f $(SQLITE_PATH) app/ml/model.pkl data/synthetic_progress.csv

docker-up: ## Start the Postgres + API stack via Docker Compose
	docker compose up --build

docker-down: ## Stop and remove Docker Compose services
	docker compose down

docker-seed: ## Generate synthetic data and train the model inside the api container
	docker compose exec api python data/generate_synthetic.py
	docker compose exec api python app/ml/train.py
