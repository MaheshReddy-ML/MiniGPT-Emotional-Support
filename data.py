"""Prepare emotional-support prompt/target pairs from the ESConv dataset.

The script downloads the ``thu-coai/esconv`` dataset, converts each counselor
turn into a supervised language-modeling pair, and stores the resulting samples
in ``data/train_pairs.json`` for tokenizer training and model fine-tuning.
"""

from datasets import load_dataset
import json
import os

print("Loading dataset...")

ds = load_dataset("thu-coai/esconv")

training_pairs = []

for item in ds["train"]:

    sample = json.loads(
        item["text"]
    )

    history = []

    for turn in sample["dialog"]:

        if turn["speaker"] == "usr":

            history.append(
                f"User: {turn['text']}"
            )

        else:

            strategy = turn.get(
                "strategy",
                "None"
            )

            prompt = (
                f"Emotion: {sample['emotion_type']}\n"
                f"Problem: {sample['problem_type']}\n"
                f"Strategy: {strategy}\n\n"
                + "\n".join(history)
                + "\nAssistant:"
            )

            target = turn["text"]

            training_pairs.append(
                {
                    "prompt": prompt,
                    "target": target
                }
            )

            history.append(
                f"Assistant: {target}"
            )

print(
    f"Total training pairs: {len(training_pairs)}"
)

os.makedirs(
    "data",
    exist_ok=True
)

with open(
    "data/train_pairs.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        training_pairs,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    "Saved: data/train_pairs.json"
)

print("\nExample Prompt:\n")
print(
    training_pairs[0]["prompt"]
)

print("\nExample Target:\n")
print(
    training_pairs[0]["target"]
)
