"""Decoder-only MiniGPT architecture for emotional-support text generation.

The model is a compact GPT-style transformer with token/position embeddings,
causal multi-head self-attention, feed-forward decoder blocks, layer
normalization, dropout, and tied input/output embedding weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """Token lookup table that maps token ids to dense vectors."""

    def __init__(self, vocab_size, d_model):
        """Initialize the embedding table.

        Args:
            vocab_size: Number of tokens in the tokenizer vocabulary.
            d_model: Hidden size of each token representation.
        """
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

    def forward(self, x):
        """Embed a batch of token ids."""
        return self.embedding(x)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention used inside each decoder block."""

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1
    ):
        """Create projection layers for query, key, value, and output tensors."""
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.attn_dropout = dropout

        self.q_proj = nn.Linear(
            d_model,
            d_model
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model
        )

    def forward(self, x):
        """Apply causal self-attention to a sequence of hidden states."""

        B, T, C = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out.view(
            B,
            T,
            C
        )

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise MLP that expands and projects transformer features."""

    def __init__(
        self,
        d_model,
        dropout=0.1
    ):
        """Build the GELU feed-forward network for a decoder block."""
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                d_model,
                4 * d_model
            ),
            nn.GELU(),
            nn.Linear(
                4 * d_model,
                d_model
            ),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Transform each token representation independently."""
        return self.net(x)


class DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block with residual connections."""

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1
    ):
        """Create one attention-plus-MLP decoder layer."""
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)

        self.attn = MultiHeadAttention(
            d_model,
            n_heads,
            dropout
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(
            d_model,
            dropout
        )

    def forward(self, x):
        """Run one decoder block over the hidden sequence."""

        x = x + self.attn(
            self.ln1(x)
        )

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


class MiniGPT(nn.Module):
    """Small GPT-style causal language model for supportive responses."""

    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=512,
        dropout=0.1
    ):
        """Initialize the MiniGPT transformer.

        Args:
            vocab_size: Tokenizer vocabulary size.
            d_model: Transformer hidden dimension.
            n_heads: Number of attention heads.
            n_layers: Number of decoder blocks.
            max_seq_len: Maximum supported context length.
            dropout: Dropout probability used in embeddings and blocks.
        """
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

        self.embedding_dropout = nn.Dropout(
            dropout
        )

        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(
            d_model
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.embedding.weight
        )

    def forward(self, input_ids):
        """Return next-token logits for each position in ``input_ids``."""

        B, T = input_ids.shape

        positions = torch.arange(
            T,
            device=input_ids.device
        ).unsqueeze(0)

        tok_emb = self.token_embedding(
            input_ids
        )

        pos_emb = self.position_embedding(
            positions
        )

        x = tok_emb + pos_emb

        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":

    model = MiniGPT(
        vocab_size=8000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=512
    )

    x = torch.randint(
        0,
        8000,
        (4, 128)
    )

    logits = model(x)

    print(logits.shape)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {total_params:,}"
    )
