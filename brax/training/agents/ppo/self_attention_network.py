from typing import Tuple, NamedTuple, Callable, Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class TransformerBlock(nn.Module):
    """Encoder block: MHSA + MLP, each with pre-LN + residual."""
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.0  # set >0 only if you want dropout & RNG plumbing

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,          # [B, T, d_model]
        *,
        mask: jnp.ndarray,       # [1, 1, T, T] or broadcastable
        train: bool,
    ) -> jnp.ndarray:
        # --- Multi-head self-attention (pre-LN) ---
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
        )(y, y, y, mask=mask)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        x = x + y  # residual

        # --- Position-wise MLP (pre-LN) ---
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.d_ff)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(self.d_model)(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        x = x + y  # residual

        return x


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> jnp.ndarray:
    """Returns [seq_len, d_model] sinusoidal positional encodings."""
    position = jnp.arange(seq_len)[:, None]              # [T, 1]
    div_term = jnp.exp(
        jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model)
    )  # [d_model/2]

    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe

class PrivilegedStateTransformer(nn.Module):
    """
    Transformer that maps a sequence of inputs (states,
    optionally concatenated with actions) to privileged state estimates.

    Inputs
    ------
    x: [B, T, d_in]
       d_in may be state_dim or state_dim + action_dim.

    Outputs
    -------
    priv_all:  [B, T, priv_dim]   (per-timestep estimates)
    priv_last: [B, priv_dim]      (estimate for last timestep, for PPO input)
    """
    priv_dim: int
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 256
    num_layers: int = 2
    max_len: int = 128
    dropout_rate: float = 0.0      # keep 0.0 to avoid RNG plumbing
    head_hidden_dim: int = 128     # Linear → activation → Linear head

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,           # [B, T, d_in]
        *,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        B, T, d_in = x.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} > max_len {self.max_len}")

        # --- Input projection to d_model ---
        x = nn.Dense(self.d_model, name="input_proj")(x)  # [B, T, d_model]

        # --- Positional encoding (sinusoidal, no params) ---
        pe = sinusoidal_positional_encoding(self.max_len, self.d_model)  # [max_len, d_model]
        pe = pe[None, :T, :]                                            # [1, T, d_model]
        x = x + pe                                                      # [B, T, d_model]

        # --- Causal mask: allow attending to self and past, not future ---
        # shape: [1, 1, T, T] so it broadcasts over batch and heads
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, T, T]

        # --- Transformer encoder stack ---
        for i in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )(x, mask=causal_mask, train=train)  # [B, T, d_model]

        x = nn.LayerNorm(name="final_layernorm")(x)  # [B, T, d_model]

        # --- Output head: safer option (Linear → activation → Linear) ---
        h_head = nn.Dense(self.head_hidden_dim, name="head_hidden")(x)
        h_head = nn.gelu(h_head)
        priv_all = nn.Dense(self.priv_dim, name="priv_out")(h_head)  # [B, T, priv_dim]

        # Use last timestep as the summary
        priv_last = priv_all[:, -1, :]  # [B, priv_dim]

        return priv_all, priv_last
