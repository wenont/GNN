.DEFAULT_GOAL := main

PYTHON = /opt/miniconda3/envs/bt/bin/python

install:
	@if conda env list | grep -q '^bt'; then \
			echo "environment exists"; \
	else \
			conda env create -f environment.yaml -n bt; \
	fi

main:
	@${PYTHON} main.py

p:
	@${PYTHON} main.py --function parameter --dataset small_molecules --verbose

p1:
	@${PYTHON} main.py --function parameter --dataset bioinformatics --verbose

p2:
	@${PYTHON} main.py --function parameter --dataset computer_vision --verbose

p3:
	@${PYTHON} main.py --function parameter --dataset social_networks --verbose

t:
	@${PYTHON} main.py --function parameter --dataset test_dataset --verbose

