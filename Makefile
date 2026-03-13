.PHONY: help install sync setup-env run worker test

UV ?= uv
PYTHON ?= python

help:
	@printf "Available targets:\n"
	@printf "  make install    - sync project dependencies with uv\n"
	@printf "  make sync       - alias for install\n"
	@printf "  make setup-env  - create .env from .env.example if missing\n"
	@printf "  make run        - run the video worker\n"
	@printf "  make worker     - alias for run\n"
	@printf "  make test       - run test suite\n"

install:
	$(UV) sync

sync: install

setup-env:
	@test -f .env || cp .env.example .env

run:
	$(UV) run $(PYTHON) -m aqshara_video_worker.run_job

worker: run

test:
	$(UV) run pytest
