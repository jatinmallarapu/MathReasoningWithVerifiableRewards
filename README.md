# Tiny Reasoner GRPO

[![CI](https://github.com/jatinmallarapu/MathReasoningWithVerifiableRewards/actions/workflows/ci.yml/badge.svg)](https://github.com/jatinmallarapu/MathReasoningWithVerifiableRewards/actions/workflows/ci.yml)

Compare supervised fine-tuning and Group Relative Policy Optimization on a small Qwen model using verifiable math rewards.

| Checkpoint | GSM8K test protocol | Accuracy |
| --- | --- | --- |
| Qwen2.5-1.5B-Instruct | Greedy | Pending GPU run |
| SFT | Greedy | Pending GPU run |
| SFT followed by GRPO | Greedy | Pending GPU run |
| SFT followed by GRPO | Eight-sample majority vote | Pending GPU run |

Use Linux, Python 3.11 and a BF16 CUDA GPU. Install with `python -m pip install -r requirements.txt`.

```bash
make train REVISION=MODEL_COMMIT_SHA
make eval REVISION=MODEL_COMMIT_SHA
python plot.py runs/grpo/history.json runs/reward-and-length.png
```

SFT trains on worked solutions. GRPO inherits the SFT adapter and samples four completions per question. A fraction-based checker gives one point for the correct numeric answer and the format reward adds 0.25 for a complete tagged response. The test set is held out of both training stages. The evaluation JSON records exact-match decisions and completions for failure analysis. Unparseable samples receive no vote. Ties go to the answer seen first.

Compare train reward and completion length against the corresponding keys in `runs/grpo/history.json`. Enable W&B explicitly with `--report-to wandb`. Report sample count alongside accuracy to make the cost of test-time sampling visible. MATH-500 evaluation needs a separate symbolic checker for nonnumeric answers.

The reward tests cover equivalent fractions and decimals, invalid fractions, and well-formed wrong answers. A wrong answer can earn the format bonus while earning zero correctness reward. Track correctness separately from total reward to catch that failure mode when comparing SFT and GRPO.

The experiment illustrates verifiable rewards and sampling at inference. Public o1 descriptions establish only the broad reasoning and test-time-compute connection. [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k) · [TRL GRPO](https://huggingface.co/docs/trl/v0.18.2/grpo_trainer) · [OpenAI reasoning research](https://openai.com/index/learning-to-reason-with-llms/)
