REVISION ?=

train:
	python experiment.py sft --revision "$(REVISION)" --output runs/sft
	python experiment.py grpo --revision "$(REVISION)" --adapter runs/sft --output runs/grpo

eval:
	python experiment.py eval --revision "$(REVISION)" --output runs/base.json
	python experiment.py eval --revision "$(REVISION)" --adapter runs/sft --output runs/sft.json
	python experiment.py eval --revision "$(REVISION)" --adapter runs/grpo --output runs/grpo.json
	python experiment.py eval --revision "$(REVISION)" --adapter runs/grpo --samples 8 --output runs/grpo-vote8.json
