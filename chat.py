"""
Interactive terminal chat for the emotional-support MiniGPT model.

Loads tokenizer.json and checkpoints/best_model.pt,
then runs a multi-turn emotional-support chat session.
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

TEMPERATURE = 0.4

TOP_K = 15

MAX_NEW_TOKENS = 50

REPETITION_PENALTY = 1.15

STRATEGY = "Reflection of feelings"


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
# Initial Context
# ==========================================================

emotion = input(
    "Emotion: "
)

problem = input(
    "Problem: "
)

history = (
    f"Emotion: {emotion}\n"
    f"Problem: {problem}\n"
    f"Strategy: {STRATEGY}\n\n"
)

print("\nChat Started")
print("Type 'quit' to exit\n")


# ==========================================================
# Chat Loop
# ==========================================================

while True:

    user_text = input(
        "You: "
    )

    if (
        user_text.lower()
        == "quit"
    ):
        break

    # ------------------------------------------------------
    # Store user message
    # ------------------------------------------------------

    history += (
        f"User: {user_text}\n"
    )

    # ------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------

    prompt = (
        history
        + "Assistant:\n"
        + "<RESPONSE>\n"
    )

    words = prompt.split()

    if len(words) > 220:

        prompt = " ".join(
            words[-220:]
        )

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

    prompt_length = (
        input_ids.shape[1]
    )

    # ------------------------------------------------------
    # Generation
    # ------------------------------------------------------

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

            # Repetition penalty

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

            # Top-K

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

            # EOS stopping

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

    # ------------------------------------------------------
    # Decode ONLY generated tokens
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------

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
            "I understand. "
            "Could you tell me a little more about that?"
        )

    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    print(
        f"\nBot: {reply}\n"
    )

    # ------------------------------------------------------
    # Save assistant reply
    # ------------------------------------------------------

    history += (
        f"Assistant: {reply}\n"
    )