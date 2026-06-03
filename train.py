"""
Train the emotional-support MiniGPT language model.

The training loop reads prompt/target pairs from data/train_pairs.json,
tokenizes them through EmotionalSupportDataset, optimizes the decoder-only
MiniGPT model with next-token cross entropy, and writes epoch/best checkpoints
to the checkpoints directory.
"""

import os

import torch
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    random_split
)

from tqdm import tqdm

from dataset import EmotionalSupportDataset
from model import MiniGPT


# ==========================================================
# Configuration
# ==========================================================

BATCH_SIZE = 32

MAX_LEN = 128

EPOCHS = 3

LR = 3e-4

VOCAB_SIZE = 16000


# ==========================================================
# Device
# ==========================================================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(
    f"Using device: {DEVICE}"
)


# ==========================================================
# Checkpoints
# ==========================================================

os.makedirs(
    "checkpoints",
    exist_ok=True
)


# ==========================================================
# Dataset
# ==========================================================

dataset = EmotionalSupportDataset(
    data_path="data/train_pairs.json",
    tokenizer_path="tokenizer.json",
    max_length=MAX_LEN
)

print(
    f"Dataset Size: {len(dataset)}"
)

train_size = int(
    0.9 * len(dataset)
)

val_size = (
    len(dataset)
    - train_size
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print(
    f"Train Size: {len(train_dataset)}"
)

print(
    f"Validation Size: {len(val_dataset)}"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)


# ==========================================================
# Model
# ==========================================================

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    d_model=256,
    n_heads=8,
    n_layers=6,
    max_seq_len=512
)

model = model.to(
    DEVICE
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {total_params:,}"
)


# ==========================================================
# Optimizer
# ==========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ==========================================================
# Training
# ==========================================================

best_val_loss = float(
    "inf"
)

train_loss_history = []

val_loss_history = []


for epoch in range(EPOCHS):

    # ======================================================
    # Train
    # ======================================================

    model.train()

    running_loss = 0.0

    valid_batches = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
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
            labels[:, 1:].reshape(
                -1
            ),
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

    train_loss = (
        running_loss
        / max(valid_batches, 1)
    )

    train_loss_history.append(
        train_loss
    )

    print(
        f"\nTrain Loss: {train_loss:.4f}"
    )

    # ======================================================
    # Validation
    # ======================================================

    model.eval()

    val_running_loss = 0.0

    val_batches = 0

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            labels = batch[
                "labels"
            ].to(DEVICE)

            logits = model(
                input_ids
            )

            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(
                    -1,
                    VOCAB_SIZE
                ),
                labels[:, 1:].reshape(
                    -1
                ),
                ignore_index=-100
            )

            val_running_loss += (
                loss.item()
            )

            val_batches += 1

    val_loss = (
        val_running_loss
        / max(val_batches, 1)
    )

    val_loss_history.append(
        val_loss
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    # ======================================================
    # Scheduler
    # ======================================================

    scheduler.step()

    current_lr = (
        scheduler.get_last_lr()[0]
    )

    print(
        f"Learning Rate: {current_lr:.8f}"
    )

    # ======================================================
    # Save Epoch Checkpoint
    # ======================================================

    epoch_path = (
        f"checkpoints/model_epoch_{epoch + 1}.pt"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        },
        epoch_path
    )

    print(
        f"Checkpoint saved: {epoch_path}"
    )

    # ======================================================
    # Best Model
    # ======================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            "checkpoints/best_model.pt"
        )

        print(
            "New best model saved."
        )


# ==========================================================
# Save Stats
# ==========================================================

torch.save(
    {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history
    },
    "checkpoints/training_stats.pt"
)

print(
    "\nTraining Complete!"
)

print(
    f"Best Validation Loss: {best_val_loss:.4f}"
)