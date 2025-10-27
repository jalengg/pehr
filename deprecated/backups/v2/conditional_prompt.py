"""
Conditional prompt modules for demographic conditioning.
Converts continuous (age) and categorical (gender, ethnicity) features to embeddings.
"""
import torch
import torch.nn as nn
from typing import Optional


class NumericalConditionalPrompt(nn.Module):
    """Embeds continuous numerical features (e.g., age) into prompt embeddings."""

    def __init__(self, n_num_features: int, hidden_dim: int, prompt_length: int = 1):
        """Initialize numerical prompt encoder.

        Args:
            n_num_features: Number of continuous features (1 for age only).
            hidden_dim: Hidden dimension size (768 for BART-base).
            prompt_length: Number of prompt vectors per feature (default 1).
        """
        super().__init__()
        self.n_num_features = n_num_features
        self.hidden_dim = hidden_dim
        self.prompt_length = prompt_length

        self.embedding = nn.Linear(n_num_features, hidden_dim * prompt_length)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """Embed numerical features.

        Args:
            x_num: [batch, n_num_features] continuous values.

        Returns:
            [batch, prompt_length * n_num_features, hidden_dim] embeddings.
        """
        batch_size = x_num.shape[0]

        embeds = self.embedding(x_num)
        embeds = embeds.reshape(batch_size, self.prompt_length * self.n_num_features, self.hidden_dim)

        return embeds


class CategoricalConditionalPrompt(nn.Module):
    """Embeds categorical features (e.g., gender, ethnicity) into prompt embeddings."""

    def __init__(self, cat_cardinalities: list[int], hidden_dim: int, prompt_length: int = 1):
        """Initialize categorical prompt encoder.

        Args:
            cat_cardinalities: List of category counts for each feature.
                              [2, 6] for gender (2 options) and ethnicity (6 options).
            hidden_dim: Hidden dimension size (768 for BART-base).
            prompt_length: Number of prompt vectors per feature (default 1).
        """
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.hidden_dim = hidden_dim
        self.prompt_length = prompt_length

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, hidden_dim * prompt_length)
            for card in cat_cardinalities
        ])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Embed categorical features.

        Args:
            x_cat: [batch, n_cat_features] categorical IDs.

        Returns:
            [batch, prompt_length * n_cat_features, hidden_dim] embeddings.
        """
        batch_size = x_cat.shape[0]
        n_cat_features = len(self.cat_cardinalities)

        cat_embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            embed = emb_layer(x_cat[:, i])
            embed = embed.reshape(batch_size, self.prompt_length, self.hidden_dim)
            cat_embeds.append(embed)

        embeds = torch.cat(cat_embeds, dim=1)
        return embeds


class ConditionalPrompt(nn.Module):
    """Combined prompt encoder for both numerical and categorical features."""

    def __init__(
        self,
        n_num_features: Optional[int] = None,
        cat_cardinalities: Optional[list[int]] = None,
        hidden_dim: int = 768,
        prompt_length: int = 1
    ):
        """Initialize combined prompt encoder.

        Args:
            n_num_features: Number of continuous features (None to disable).
            cat_cardinalities: Category counts for each categorical feature (None to disable).
            hidden_dim: Hidden dimension size (768 for BART-base).
            prompt_length: Number of prompt vectors per feature.
        """
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.hidden_dim = hidden_dim
        self.prompt_length = prompt_length

        if n_num_features is not None and n_num_features > 0:
            self.num_prompt = NumericalConditionalPrompt(n_num_features, hidden_dim, prompt_length)
        else:
            self.num_prompt = None

        if cat_cardinalities is not None and len(cat_cardinalities) > 0:
            self.cat_prompt = CategoricalConditionalPrompt(cat_cardinalities, hidden_dim, prompt_length)
        else:
            self.cat_prompt = None

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode demographics to prompt embeddings.

        Args:
            x_num: [batch, n_num_features] continuous values (optional).
            x_cat: [batch, n_cat_features] categorical IDs (optional).

        Returns:
            [batch, total_prompts, hidden_dim] combined prompt embeddings.
        """
        prompts = []

        if x_num is not None and self.num_prompt is not None:
            num_embeds = self.num_prompt(x_num)
            prompts.append(num_embeds)

        if x_cat is not None and self.cat_prompt is not None:
            cat_embeds = self.cat_prompt(x_cat)
            prompts.append(cat_embeds)

        if len(prompts) == 0:
            raise ValueError("No prompt embeddings generated. Provide x_num or x_cat.")

        combined_prompts = torch.cat(prompts, dim=1)
        return combined_prompts

    def get_num_prompts(self) -> int:
        """Calculate total number of prompt tokens."""
        num_prompts = 0

        if self.num_prompt is not None:
            num_prompts += self.n_num_features * self.prompt_length

        if self.cat_prompt is not None:
            num_prompts += len(self.cat_cardinalities) * self.prompt_length

        return num_prompts
