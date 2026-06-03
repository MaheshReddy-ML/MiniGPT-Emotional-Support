"""Single-prompt inference demo for the emotional-support MiniGPT model.

The script loads ``tokenizer.json`` and ``checkpoints/best_model.pt``, builds a
fixed emotional-support prompt, samples a short answer, and prints only the
generated assistant reply. Use this as a fast smoke test before running the
interactive chat script.
"""

import torch

from tokenizers import Tokenizer

from model import MiniGPT


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

VOCAB_SIZE = 8000

MAX_NEW_TOKENS = 60

TEMPERATURE = 0.8

TOP_K = 30

REPETITION_PENALTY = 1.2


tokenizer = Tokenizer.from_file(
    "tokenizer.json"
)

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    d_model=256,
    n_heads=8,
    n_layers=6,
    max_seq_len=512
)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pt",
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()


prompt = """
Emotion: anxiety
Problem: job crisis
Strategy: Reflection of feelings

User: I am afraid I might lose my job.
Assistant:

<RESPONSE>
"""

prompt = prompt.replace(
    "\n",
    " "
)

encoding = tokenizer.encode(
    prompt
)

input_ids = torch.tensor(
    [encoding.ids],
    dtype=torch.long,
    device=DEVICE
)


with torch.no_grad():

    for _ in range(MAX_NEW_TOKENS):

        logits = model(
            input_ids
        )

        next_token_logits = logits[
            :,
            -1,
            :
        ]

        for token_id in set(
            input_ids[0].tolist()
        ):
            next_token_logits[
                0,
                token_id
            ] /= REPETITION_PENALTY

        next_token_logits = (
            next_token_logits
            / TEMPERATURE
        )

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

        input_ids = torch.cat(
            [
                input_ids,
                next_token
            ],
            dim=1
        )

        current_text = tokenizer.decode(
            input_ids[0]
            .cpu()
            .tolist()
        )

        generated_part = current_text[
            len(prompt):
        ]

        if (
            "User:" in generated_part
            or "Emotion:" in generated_part
            or "Problem:" in generated_part
            or "Strategy:" in generated_part
        ):
            break


generated_text = tokenizer.decode(
    input_ids[0]
    .cpu()
    .tolist()
)

reply = generated_text[
    len(prompt):
]

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

print("\n")
print("=" * 80)
print(reply)
print("=" * 80)
