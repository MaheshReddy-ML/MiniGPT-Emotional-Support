"""Interactive terminal chat for the emotional-support MiniGPT model.

This script loads the trained tokenizer and best checkpoint, collects an
emotion/problem pair from the user, and then runs a simple multi-turn chat loop.
Responses are sampled with temperature, top-k filtering, and a repetition
penalty so the tiny model can produce more varied supportive text.
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

TEMPERATURE = 0.8
TOP_K = 30
MAX_NEW_TOKENS = 60

REPETITION_PENALTY = 1.2

STRATEGY = "Reflection of feelings"


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


emotion = input(
    "Emotion: "
)

problem = input(
    "Problem: "
)

history = (
    f"Emotion: {emotion} "
    f"Problem: {problem} "
    f"Strategy: {STRATEGY} "
)

print("\nChat Started")
print("Type 'quit' to exit\n")


while True:

    user_text = input("You: ")

    if user_text.lower() == "quit":
        break

    history += (
        f"User: {user_text} "
        f"Assistant: "
        f"<RESPONSE> "
    )

    # Keep context close to training distribution
    words = history.split()

    if len(words) > 120:
        history = " ".join(
            words[-120:]
        )

    encoding = tokenizer.encode(
        history
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

            # repetition penalty
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

            topk_values, topk_indices = (
                torch.topk(
                    next_token_logits,
                    TOP_K
                )
            )

            probs = torch.softmax(
                topk_values,
                dim=-1
            )

            sampled_index = (
                torch.multinomial(
                    probs,
                    num_samples=1
                )
            )

            next_token = (
                topk_indices.gather(
                    -1,
                    sampled_index
                )
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
                len(history):
            ]

            # Stop if model starts a new section
            if (
                "User:" in generated_part
                or "Assistant:" in generated_part
                or "Emotion:" in generated_part
                or "Problem:" in generated_part
                or "Strategy:" in generated_part
                or "<RESPONSE>" in generated_part
            ):
                break

            # Soft token-length limit
            generated_tokens = tokenizer.encode(
                generated_part
            ).ids

            if len(generated_tokens) > 50:
                break

    generated_text = tokenizer.decode(
        input_ids[0]
        .cpu()
        .tolist()
    )

    reply = generated_text[
        len(history):
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

    if not reply:
        reply = (
            "I understand. "
            "Could you tell me a little more about that?"
        )

    print(f"\nBot: {reply}\n")

    history += (
        reply + " "
    )
