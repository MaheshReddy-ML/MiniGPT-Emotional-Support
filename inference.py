"""
Single-prompt inference demo for the emotional-support MiniGPT model.

The script loads tokenizer.json and checkpoints/best_model.pt,
builds a fixed emotional-support prompt, samples a short answer,
and prints only the generated assistant reply.
"""

import torch

from tokenizers import Tokenizer

from model import MiniGPT


# ==========================================================
# Config
# ==========================================================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

VOCAB_SIZE = 16000

MAX_NEW_TOKENS = 100

TEMPERATURE = 0.6

TOP_K = 20

REPETITION_PENALTY = 1.15


# ==========================================================
# Load Tokenizer
# ==========================================================

tokenizer = Tokenizer.from_file(
    "tokenizer.json"
)

bos_id = tokenizer.token_to_id(
    "[BOS]"
)

eos_id = tokenizer.token_to_id(
    "[EOS]"
)


# ==========================================================
# Load Model
# ==========================================================

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    d_model=256,
    n_heads=8,
    n_layers=6,
    max_seq_len=512
)

checkpoint = torch.load(
    "checkpoints/best_model.pt",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

model.to(
    DEVICE
)

model.eval()


# ==========================================================
# Prompt
# ==========================================================

prompt = """
Emotion: anxiety
Problem: studies
Strategy: Reflection of feelings

User: I failed my exam and I feel like a complete failure.
Assistant:

<RESPONSE>
"""

prompt = prompt.strip()

encoding = tokenizer.encode(
    prompt
)

input_ids = torch.tensor(
    [
        [bos_id]
        + encoding.ids
    ],
    dtype=torch.long,
    device=DEVICE
)

# IMPORTANT
# Save prompt length BEFORE generation

prompt_length = input_ids.shape[1]


# ==========================================================
# Generation
# ==========================================================

with torch.no_grad():

    for _ in range(
        MAX_NEW_TOKENS
    ):

        logits = model(
            input_ids
        )

        next_token_logits = logits[
            :,
            -1,
            :
        ]

        # Repetition Penalty

        for token_id in set(
            input_ids[0].tolist()
        ):

            next_token_logits[
                0,
                token_id
            ] /= REPETITION_PENALTY

        # Temperature

        next_token_logits = (
            next_token_logits
            / TEMPERATURE
        )

        # Top-K Sampling

        topk_values, topk_indices = torch.topk(
            next_token_logits,
            TOP_K
        )

        probs = torch.softmax(
            topk_values,
            dim=-1
        )

        sampled_index = torch.multinomial(
            probs,
            num_samples=1
        )

        next_token = topk_indices.gather(
            -1,
            sampled_index
        )

        # Stop at EOS

        if (
            next_token.item()
            == eos_id
        ):
            break

        input_ids = torch.cat(
            [
                input_ids,
                next_token
            ],
            dim=1
        )


# ==========================================================
# Decode ONLY Generated Tokens
# ==========================================================

generated_ids = (
    input_ids[0]
    .cpu()
    .tolist()
)

reply_ids = generated_ids[
    prompt_length:
]

reply = tokenizer.decode(
    reply_ids
)

# Remove accidental special sections

for stop_token in [
    "User:",
    "Assistant:",
    "Emotion:",
    "Problem:",
    "Strategy:",
    "<RESPONSE>"
]:

    if stop_token in reply:

        reply = reply.split(
            stop_token
        )[0]

reply = reply.strip()

if len(reply) == 0:

    reply = (
        "No response generated."
    )


# ==========================================================
# Output
# ==========================================================

print("\n")
print("=" * 80)
print("MODEL RESPONSE:\n")
print(reply)
print("=" * 80)