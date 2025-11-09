import torch
from torch import nn
import math
from typing import Dict, Optional

class MILConjunctivePooling(nn.Module):
    """Conjunctive MIL pooling. Instance attention then weighting of instance predictions."""

    def __init__(
        self,
        d_in: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
        aggregation_func: str = "mean"
    ):
        super().__init__()

        self.d_in = d_in
        self.dropout_p = dropout
        self.apply_positional_encoding = apply_positional_encoding
        
        if apply_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_in)
        
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        
        self.aggregation_func = aggregation_func
        
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, 1)
        
        if aggregation_func == "logavgexp":
            self.t = nn.Parameter(torch.ones((1,))) # Learnable temperature parameter
        else:
            self.register_parameter("t", None)

        
    def aggregate(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if self.aggregation_func == "mean":
            return torch.mean(x, dim=dim)
        elif self.aggregation_func == "logavgexp":
            # Optimize log of temperature and prevent it from zeroing
            t = self.t.exp().clamp(min = 1e-8)
            lse = torch.logsumexp(x / t, dim=dim)
            return t * (lse - math.log(x.size(dim)))
        else:
            raise ValueError(f"Unknown aggregation function: {self.aggregation_func}")

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :return: Dictionary containing bag_logits, interpretation (instance predictions weight by attention),
        unweighted instance logits, and attn values.
        """
        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        instance_embeddings = instance_embeddings.transpose(1, 2)
        # Add positional encodings
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        # Calculate attention
        attn = self.attention_head(instance_embeddings)
        # Classify instances
        instance_logits = self.instance_classifier(instance_embeddings)
        # Weight and sum
        weighted_instance_logits = instance_logits * attn
        
        bag_logits = self.aggregate(weighted_instance_logits, dim=1)
        
        return {
            "bag_logits": bag_logits,
            "interpretation": weighted_instance_logits.transpose(1, 2),
            # Also return additional outputs
            "instance_logits": instance_logits.transpose(1, 2),
            "attn": attn,
        }


class PositionalEncoding(nn.Module):
    """
    Adapted from (under BSD 3-Clause License):
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Batch, ts len, d_model
        self.pe = torch.zeros(1, max_len, d_model)
        self.pe[0, :, 0::2] = torch.sin(position * div_term) # for even embedding entries
        self.pe[0, :, 1::2] = torch.cos(position * div_term) # for odd embedding entries
        # self.register_buffer("pe", self.pe)

    def forward(self, x: torch.Tensor, x_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply positional encoding to a set of time series embeddings.

        :param x: Embeddings.
        :param x_pos: Optional positions (indices) of each timestep. If not provided, will use range(len(time series)),
        i.e. 0,...,t-1.
        :return: A tensor the same shape as x, but with positional encoding added to it.
        """
        
        if x_pos is None:
            x_pe = self.pe[:, : x.size(1)]
        else:
            x_pe = self.pe[0, x_pos]
        x = x + x_pe
        return x