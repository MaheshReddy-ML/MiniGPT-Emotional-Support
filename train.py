"""Train the emotional-support MiniGPT language model.

The training loop reads prompt/target pairs from ``data/train_pairs.json``,
tokenizes them through ``EmotionalSupportDataset``, optimizes the decoder-only
MiniGPT model with next-token cross entropy, and writes epoch/best checkpoints
to the ``checkpoints`` directory.
"""

import os

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmotionalSupportDataset
from model import MiniGPT


BATCH_SIZE = 16
MAX_LEN = 128

EPOCHS = 5

LR = 3e-4

VOCAB_SIZE = 8000


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")

os.makedirs(
    "checkpoints",
    exist_ok=True
)

dataset = EmotionalSupportDataset(
    data_path="data/train_pairs.json",
    tokenizer_path="tokenizer.json",
    max_length=MAX_LEN
)

print(
    f"Dataset Size: {len(dataset)}"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    d_model=256,
    n_heads=8,
    n_layers=6,
    max_seq_len=512
)

model = model.to(DEVICE)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {total_params:,}"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.01
)

best_loss = float("inf")

loss_history = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    valid_batches = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for batch in pbar:

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            input_ids
        )

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(
                -1,
                VOCAB_SIZE
            ),
            labels[:, 1:].reshape(-1),
            ignore_index=-100
        )

        if torch.isnan(loss):

            print(
                "Skipping NaN batch"
            )

            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        valid_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = (
        running_loss
        / max(valid_batches, 1)
    )

    loss_history.append(
        avg_loss
    )

    print(
        f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}"
    )

    epoch_path = (
        f"checkpoints/model_epoch_{epoch+1}.pt"
    )

    torch.save(
        model.state_dict(),
        epoch_path
    )

    print(
        f"Checkpoint saved: {epoch_path}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pt"
        )

        print(
            "New best model saved."
        )

torch.save(
    {
        "loss_history": loss_history
    },
    "checkpoints/training_stats.pt"
)

print("\nTraining Complete!")

print(
    f"Best Loss: {best_loss:.4f}"
)
print(
    f"Valid batches: {valid_batches}"
)
