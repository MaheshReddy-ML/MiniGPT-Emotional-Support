"""
Decoder-only MiniGPT architecture for emotional-support text generation.

V2 Improvements:
- RMSNorm
- SwiGLU FeedForward
- Weight Tying
- Better Weight Initialization
- Flash Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

    def forward(
        self,
        x
    ):

        return self.embedding(
            x
        )


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1
    ):

        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads

        self.head_dim = (
            d_model // n_heads
        )

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

    def forward(
        self,
        x
    ):

        B, T, C = x.shape

        Q = self.q_proj(x)

        K = self.k_proj(x)

        V = self.v_proj(x)

        Q = Q.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        K = K.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        V = V.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=(
                self.attn_dropout
                if self.training
                else 0.0
            ),
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

        return self.out_proj(
            out
        )


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1
    ):

        super().__init__()

        hidden_dim = int(
            8 * d_model / 3
        )

        self.gate_proj = nn.Linear(
            d_model,
            hidden_dim
        )

        self.up_proj = nn.Linear(
            d_model,
            hidden_dim
        )

        self.down_proj = nn.Linear(
            hidden_dim,
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x
    ):

        x = (
            F.silu(
                self.gate_proj(x)
            )
            *
            self.up_proj(x)
        )

        x = self.down_proj(
            x
        )

        return self.dropout(
            x
        )


class DecoderBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1
    ):

        super().__init__()

        self.ln1 = nn.RMSNorm(
            d_model
        )

        self.attn = MultiHeadAttention(
            d_model,
            n_heads,
            dropout
        )

        self.ln2 = nn.RMSNorm(
            d_model
        )

        self.ffn = FeedForward(
            d_model,
            dropout
        )

    def forward(
        self,
        x
    ):

        x = x + self.attn(
            self.ln1(x)
        )

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=512,
        dropout=0.1
    ):

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

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout
                )
                for _ in range(
                    n_layers
                )
            ]
        )

        self.ln_f = nn.RMSNorm(
            d_model
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

        # Weight Tying

        self.lm_head.weight = (
            self.token_embedding.embedding.weight
        )

        self.apply(
            self._init_weights
        )

    def _init_weights(
        self,
        module
    ):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        input_ids
    ):

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

        x = self.embedding_dropout(
            x
        )

        for block in self.blocks:

            x = block(
                x
            )

        x = self.ln_f(
            x
        )

        logits = self.lm_head(
            x
        )

        return logits


if __name__ == "__main__":

    model = MiniGPT(
        vocab_size=16000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=512
    )

    x = torch.randint(
        0,
        16000,
        (4, 128)
    )

    logits = model(
        x
    )

    print(
        logits.shape
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {total_params:,}"
    )