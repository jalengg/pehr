"""
Complete PromptBART model for EHR generation with demographic conditioning.
Combines ConditionalPrompt, PromptBartEncoder, and PromptBartDecoder.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import BartConfig, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from conditional_prompt import ConditionalPrompt
from prompt_bart_encoder import PromptBartEncoder
from prompt_bart_decoder import PromptBartDecoder


class PromptBartModel(BartForConditionalGeneration):
    """BART model with demographic prompt conditioning for EHR generation."""

    def __init__(
        self,
        config: BartConfig,
        n_num_features: Optional[int] = None,
        cat_cardinalities: Optional[list[int]] = None,
        d_hidden: int = 128,
        prompt_length: int = 1
    ):
        """Initialize PromptBART model with dual prompt conditioning.

        Args:
            config: BART configuration.
            n_num_features: Number of continuous features (e.g., 1 for age).
            cat_cardinalities: Category counts for categorical features [n_genders, n_ethnicities].
            d_hidden: Intermediate reparameterization dimension (default 128).
            prompt_length: Number of prompt vectors per feature.
        """
        super().__init__(config)

        # Replace encoder and decoder with prompt-aware versions
        self.model.encoder = PromptBartEncoder(config, self.model.shared)
        self.model.decoder = PromptBartDecoder(config, self.model.shared)

        # Add SEPARATE conditional prompt encoders for encoder and decoder
        # This provides stronger demographic conditioning than shared prompts
        if n_num_features is not None or cat_cardinalities is not None:
            # Encoder prompt encoder
            self.encoder_prompt_encoder = ConditionalPrompt(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                hidden_dim=config.d_model,
                d_hidden=d_hidden,
                prompt_length=prompt_length
            )
            # Decoder prompt encoder (separate parameters)
            self.decoder_prompt_encoder = ConditionalPrompt(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                hidden_dim=config.d_model,
                d_hidden=d_hidden,
                prompt_length=prompt_length
            )
            self.num_prompts = self.encoder_prompt_encoder.get_num_prompts()
        else:
            self.encoder_prompt_encoder = None
            self.decoder_prompt_encoder = None
            self.num_prompts = 0

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        x_num: Optional[torch.FloatTensor] = None,
        x_cat: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        """Forward pass with demographic conditioning.

        Args:
            input_ids: [batch, seq_len] encoder input token IDs.
            attention_mask: [batch, seq_len] encoder attention mask.
            decoder_input_ids: [batch, tgt_len] decoder input token IDs.
            labels: [batch, tgt_len] target labels for loss.
            x_num: [batch, n_num_features] continuous demographic features.
            x_cat: [batch, n_cat_features] categorical demographic features.
            Other args: Standard BART arguments.

        Returns:
            Seq2SeqLMOutput with loss, logits, and hidden states.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode demographic prompts separately for encoder and decoder
        # Only prepend prompts on first step (when no cache exists)
        encoder_prompt_embeds = None
        decoder_prompt_embeds = None
        if (x_num is not None or x_cat is not None) and past_key_values is None:
            if self.encoder_prompt_encoder is not None:
                encoder_prompt_embeds = self.encoder_prompt_encoder(x_num=x_num, x_cat=x_cat)
            if self.decoder_prompt_encoder is not None:
                decoder_prompt_embeds = self.decoder_prompt_encoder(x_num=x_num, x_cat=x_cat)

        # Prepare decoder input IDs
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Encoder forward pass (with encoder prompts)
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                inputs_prompt_embeds=encoder_prompt_embeds,  # Encoder-specific prompts
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Extend encoder attention mask for prompts
        encoder_attention_mask = attention_mask
        if encoder_prompt_embeds is not None and attention_mask is not None:
            batch_size, n_prompts = encoder_prompt_embeds.shape[:2]
            prompt_mask = torch.ones(batch_size, n_prompts, dtype=attention_mask.dtype, device=attention_mask.device)
            encoder_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Decoder forward pass (with decoder prompts)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            inputs_prompt_embeds=decoder_prompt_embeds,  # Decoder-specific prompts
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Language modeling head
        lm_logits = self.lm_head(decoder_outputs[0])

        # If decoder prompts were prepended, slice them off before loss computation
        if decoder_prompt_embeds is not None and labels is not None:
            # decoder_outputs[0] shape: [batch, n_prompts + seq_len, hidden_dim]
            # We only want logits for the actual sequence positions
            n_prompts = decoder_prompt_embeds.shape[1]
            lm_logits = lm_logits[:, n_prompts:, :]  # Remove prompt positions

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        x_num=None,
        x_cat=None,
        **kwargs
    ):
        """Prepare inputs for autoregressive generation."""
        # Cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "x_num": x_num,  # Pass demographics through
            "x_cat": x_cat,
        }

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids,
        expand_size=1,
        is_encoder_decoder=True,
        attention_mask=None,
        encoder_outputs=None,
        x_num=None,
        x_cat=None,
        **model_kwargs,
    ):
        """Expand inputs for beam search or multiple samples."""
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if encoder_outputs is not None:
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        # Expand demographics
        if x_num is not None:
            model_kwargs["x_num"] = x_num.index_select(0, expanded_return_idx)

        if x_cat is not None:
            model_kwargs["x_cat"] = x_cat.index_select(0, expanded_return_idx)

        return input_ids, model_kwargs


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """Shift input ids one token to the right for teacher forcing."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("config.pad_token_id must be defined for sequence generation")

    # Replace -100 in labels with pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
