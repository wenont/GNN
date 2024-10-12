.DEFAULT_GOAL := main

PYTHON = /opt/miniconda3/envs/bt/bin/python

say_hello:
	@echo "Hello, World!"

install:
	@if conda env list | grep -q '^bt'; then \
			echo "environment exists"; \
	else \
			conda env create -f environment.yaml -n bt; \
	fi

main:
	@${PYTHON} main.py

parameter:
	@${PYTHON} main.py --function parameter --dataset small_molecules.txt --verbose