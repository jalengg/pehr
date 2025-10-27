"""
Custom BART encoder that accepts prompt embeddings for conditioning.
Prepends demographic prompt embeddings to input token embeddings.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import BartModel
from transformers.models.bart.modeling_bart import BartEncoder


class PromptBartEncoder(BartEncoder):
    """BART encoder modified to accept and prepend prompt embeddings."""

    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        self.embed_scale = None
        if config.scale_embedding:
            self.embed_scale = (config.d_model ** 0.5)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass with optional prompt embeddings.

        Args:
            input_ids: [batch, seq_len] token IDs.
            attention_mask: [batch, seq_len] attention mask.
            inputs_prompt_embeds: [batch, n_prompts, hidden_dim] prompt embeddings (optional).
            Other args: Standard BART encoder arguments.

        Returns:
            BART encoder outputs with prompt-conditioned representations.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if self.embed_scale is not None:
                inputs_embeds = inputs_embeds * self.embed_scale

        # Prepend prompt embeddings if provided
        if inputs_prompt_embeds is not None:
            # Concatenate prompt embeddings before input embeddings
            # inputs_prompt_embeds: [batch, n_prompts, hidden_dim]
            # inputs_embeds: [batch, seq_len, hidden_dim]
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)
            # Result: [batch, n_prompts + seq_len, hidden_dim]

            # Extend attention mask to account for prepended prompts
            batch_size, n_prompts = inputs_prompt_embeds.shape[:2]
            prompt_attention_mask = torch.ones(
                batch_size, n_prompts,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )

            if attention_mask is not None:
                attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            else:
                # Create full attention mask if none provided
                seq_len = inputs_embeds.shape[1] - n_prompts
                seq_attention_mask = torch.ones(
                    batch_size, seq_len,
                    dtype=prompt_attention_mask.dtype,
                    device=prompt_attention_mask.device
                )
                attention_mask = torch.cat([prompt_attention_mask, seq_attention_mask], dim=1)

        # Get positional embeddings
        embed_pos = self.embed_positions(inputs_embeds)

        # Add positional embeddings
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Expand attention_mask
        if attention_mask is not None:
            # [batch, seq_len] -> [batch, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        # Pass through encoder layers
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Check if head_mask has correct number of layers
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(
                    f"head_mask should have {len(self.layers)} layers, but has {head_mask.size()[0]}"
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_head_mask = head_mask[idx] if head_mask is not None else None

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expand attention mask from [batch, src_len] to [batch, 1, tgt_len, src_len]."""
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
