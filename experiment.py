import argparse
import json
from collections import Counter
from pathlib import Path

from rewards import correctness, format_reward, number, prediction

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "Solve the problem. Put your work in <think> tags and the final number in <answer> tags."


def records(split):
    from datasets import load_dataset

    for item in load_dataset("openai/gsm8k", "main", split=split):
        solution, answer = item["answer"].rsplit("####", 1)
        yield {
            "prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": item["question"]}],
            "completion": [{"role": "assistant", "content":
                            f"<think>{solution.strip()}</think><answer>{answer.strip()}</answer>"}],
            "answer": answer.strip(),
        }


def main(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=args.revision, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=args.revision, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    peft = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=args.stage == "grpo")
        peft = None
    if args.stage != "eval":
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

        dataset = Dataset.from_list(list(records("train")))
        if args.stage == "sft":
            config = SFTConfig(
                output_dir=args.output, num_train_epochs=1, max_length=1024,
                per_device_train_batch_size=2, gradient_accumulation_steps=8,
                completion_only_loss=True, bf16=True, logging_steps=10,
                report_to=args.report_to, seed=args.seed,
            )
            trainer = SFTTrainer(model=model, args=config, train_dataset=dataset.remove_columns("answer"),
                                 processing_class=tokenizer, peft_config=peft)
        else:
            config = GRPOConfig(
                output_dir=args.output, max_steps=args.steps, max_prompt_length=512,
                max_completion_length=256, num_generations=4, per_device_train_batch_size=4,
                gradient_accumulation_steps=2, gradient_checkpointing=True, bf16=True,
                logging_steps=5, report_to=args.report_to, seed=args.seed,
            )
            trainer = GRPOTrainer(
                model=model, args=config, train_dataset=dataset.remove_columns("completion"),
                reward_funcs=[correctness, format_reward], processing_class=tokenizer, peft_config=peft,
            )
        trainer.train()
        trainer.save_model(args.output)
        Path(args.output, "history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
        return

    model.eval()
    rows = []
    for index, row in enumerate(records("test")):
        if args.limit and index >= args.limit:
            break
        inputs = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        generation = {"do_sample": args.samples > 1, "num_return_sequences": args.samples,
                      "max_new_tokens": 256}
        if args.samples > 1:
            generation.update(temperature=0.7, top_p=0.9)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation)
        decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        votes = Counter(value for text in decoded if (value := prediction(text)) is not None)
        winner = votes.most_common(1)[0][0] if votes else None
        rows.append({"row": index, "expected": row["answer"], "vote": str(winner),
                     "correct": winner is not None and winner == number(row["answer"]),
                     "completions": decoded})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"revision": args.revision, "adapter": args.adapter, "seed": args.seed,
                                  "samples": args.samples, "accuracy": sum(x["correct"] for x in rows) / len(rows),
                                  "predictions": rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sft", "grpo", "eval"])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, choices=[1, 8], default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    main(parser.parse_args())
