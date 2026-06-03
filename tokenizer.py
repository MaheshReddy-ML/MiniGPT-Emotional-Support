"""Train the BPE tokenizer used by the emotional-support MiniGPT model.

The tokenizer is trained from the prepared prompt/response pairs in
``data/train_pairs.json``. It writes a flattened corpus to
``data/tokenizer_corpus.txt`` and saves the final tokenizer configuration to
``tokenizer.json``.
"""

import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


with open(
    "data/train_pairs.json",
    "r",
    encoding="utf-8"
) as f:

    pairs = json.load(f)


with open(
    "data/tokenizer_corpus.txt",
    "w",
    encoding="utf-8"
) as f:

    for sample in pairs:

        text = (
            sample["prompt"]
            + " <RESPONSE> "
            + sample["target"]
        )

        f.write(
            text.replace("\n", " ")
            + "\n"
        )


tokenizer = Tokenizer(
    BPE(
        unk_token="[UNK]"
    )
)

tokenizer.pre_tokenizer = (
    Whitespace()
)

trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "<RESPONSE>"
    ]
)

tokenizer.train(
    ["data/tokenizer_corpus.txt"],
    trainer=trainer
)

tokenizer.save(
    "tokenizer.json"
)

print(
    "Tokenizer Saved!"
)

print(
    "Vocabulary Size:",
    tokenizer.get_vocab_size()
)


tokenizer = Tokenizer.from_file(
    "tokenizer.json"
)

text = """
Emotion: anxiety
Problem: studies
Strategy: Reflection of feelings

User: I failed my exam.

Assistant:

<RESPONSE>

That sounds really difficult.
"""

encoded = tokenizer.encode(
    text
)

print(
    encoded.tokens[:50]
)

print(
    encoded.ids[:50]
)
