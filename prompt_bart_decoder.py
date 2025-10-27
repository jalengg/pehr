"""
Custom BART decoder that accepts prompt embeddings for conditioning.
Prepends demographic prompt embeddings to decoder input embeddings.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.bart.modeling_bart import BartDecoder


class PromptBartDecoder(BartDecoder):
    """BART decoder modified to accept and prepend prompt embeddings."""

    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        self.embed_scale = None
        if config.scale_embedding:
            self.embed_scale = (config.d_model ** 0.5)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_prompt_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass with optional prompt embeddings.

        Args:
            input_ids: [batch, tgt_seq_len] decoder token IDs.
            attention_mask: [batch, tgt_seq_len] decoder attention mask.
            encoder_hidden_states: [batch, src_seq_len, hidden_dim] encoder outputs.
            encoder_attention_mask: [batch, src_seq_len] encoder attention mask.
            inputs_prompt_embeds: [batch, n_prompts, hidden_dim] prompt embeddings (optional).
            Other args: Standard BART decoder arguments.

        Returns:
            BART decoder outputs with prompt-conditioned representations.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get decoder input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if self.embed_scale is not None:
                inputs_embeds = inputs_embeds * self.embed_scale

        # Store original sequence length before prepending prompts
        original_seq_len = inputs_embeds.shape[1]

        # Prepend prompt embeddings if provided
        if inputs_prompt_embeds is not None:
            # Concatenate prompt embeddings before decoder input embeddings
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)

            # Extend attention mask for prepended prompts
            batch_size, n_prompts = inputs_prompt_embeds.shape[:2]
            prompt_attention_mask = torch.ones(
                batch_size, n_prompts,
                dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=inputs_embeds.device
            )

            if attention_mask is not None:
                attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            else:
                # Create attention mask for all tokens (prompts + decoder input)
                total_seq_len = inputs_embeds.shape[1]
                attention_mask = torch.ones(
                    batch_size, total_seq_len,
                    dtype=torch.long,
                    device=inputs_embeds.device
                )

        # Get positional embeddings for full sequence (prompts + decoder tokens)
        past_key_values_length = 0
        if past_key_values is not None:
            # Handle Cache object (new API) or tuple (old API)
            if hasattr(past_key_values, 'get_seq_length'):
                past_key_values_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
                past_key_values_length = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0

        positions = self.embed_positions(inputs_embeds, past_key_values_length)

        # Add positional embeddings
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Expand attention masks
        if attention_mask is not None:
            # Create causal mask for decoder
            combined_attention_mask = _make_causal_mask(
                inputs_embeds.shape[:2],
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
            # Combine with attention mask
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=inputs_embeds.shape[1])
            combined_attention_mask = combined_attention_mask + expanded_attn_mask
        else:
            combined_attention_mask = _make_causal_mask(
                inputs_embeds.shape[:2],
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        # Expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=inputs_embeds.shape[1])

        # Pass through decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Cache is handled by past_key_values object, not returned in tuple
        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )

        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


def _make_causal_mask(
    input_shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
):
    """Create causal mask for decoder self-attention."""
    batch_size, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expand attention mask from [batch, src_len] to [batch, 1, tgt_len, src_len]."""
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
