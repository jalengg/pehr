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


class PromptBartWithDemographicPrediction(PromptBartModel):
    """PromptBART with auxiliary age and sex prediction for medical validity.

    Extends PromptBartModel with two auxiliary prediction heads:
    1. Age predictor: Regresses continuous age from generated codes
    2. Sex predictor: Classifies binary sex from generated codes

    These auxiliary tasks create bidirectional consistency constraints,
    forcing the model to generate codes consistent with input demographics.

    See MULTITASK_LEARNING.md for full technical details.
    """

    def __init__(
        self,
        config: BartConfig,
        n_num_features: Optional[int] = 1,
        cat_cardinalities: Optional[list[int]] = None,
        d_hidden: int = 128,
        prompt_length: int = 1,
        age_loss_weight: float = 0.3,
        sex_loss_weight: float = 0.2
    ):
        """Initialize model with auxiliary prediction heads.

        Args:
            config: BART configuration.
            n_num_features: Number of continuous features (1 for age).
            cat_cardinalities: [2] for sex only (M/F), race removed.
            d_hidden: Reparameterization dimension (default 128).
            prompt_length: Number of prompt vectors per feature.
            age_loss_weight: Weight for age prediction loss (λ_age).
            sex_loss_weight: Weight for sex prediction loss (λ_sex).
        """
        if cat_cardinalities is None:
            cat_cardinalities = [2]

        super().__init__(config, n_num_features, cat_cardinalities, d_hidden, prompt_length)

        # Age prediction head (classification into 6 age brackets)
        # Age brackets: [0-2, 2-12, 12-18, 18-40, 40-65, 65+]
        self.age_predictor = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6)  # 6 age bracket classes
        )

        # Age bracket definitions
        self.age_brackets = [
            (0, 2),      # Class 0: Neonatal/infant
            (2, 12),     # Class 1: Pediatric
            (12, 18),    # Class 2: Adolescent
            (18, 40),    # Class 3: Young adult
            (40, 65),    # Class 4: Middle age
            (65, 200)    # Class 5: Elderly
        ]

        # Sex prediction head (binary classification)
        self.sex_predictor = nn.Linear(config.d_model, 2)

        # Loss weights
        self.age_loss_weight = age_loss_weight
        self.sex_loss_weight = sex_loss_weight

        # Initialize auxiliary heads
        self.age_predictor.apply(self._init_weights)
        self.sex_predictor.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for auxiliary prediction heads."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def age_to_class(self, age: torch.Tensor) -> torch.Tensor:
        """Convert continuous age to age bracket class (0-5).

        Args:
            age: Tensor of ages [batch] or scalar.

        Returns:
            Tensor of age classes [batch] or scalar.
        """
        # Handle both scalar and batched inputs
        is_scalar = age.dim() == 0
        if is_scalar:
            age = age.unsqueeze(0)

        age_classes = torch.zeros_like(age, dtype=torch.long)
        for i, (low, high) in enumerate(self.age_brackets):
            mask = (age >= low) & (age < high)
            age_classes[mask] = i

        return age_classes.squeeze(0) if is_scalar else age_classes

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
        code_offset: Optional[int] = 6,  # NEW: Token ID where diagnosis codes start
    ) -> Seq2SeqLMOutput:
        """Forward pass with auxiliary demographic prediction.

        Args:
            Standard BART arguments plus:
            x_num: [batch, 1] continuous age values.
            x_cat: [batch, 1] categorical sex IDs.
            labels: [batch, seq_len] target token IDs.

        Returns:
            Seq2SeqLMOutput with additional fields:
            - age_loss: Age prediction loss (if training)
            - sex_loss: Sex prediction loss (if training)
            - lm_loss: Language modeling loss (if training)
        """
        # Ensure output_hidden_states is True for auxiliary tasks
        output_hidden_states = True

        # Standard PromptBART forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            x_num=x_num,
            x_cat=x_cat,
        )

        # Auxiliary tasks during training only
        if self.training and x_num is not None and x_cat is not None and labels is not None:
            # Extract decoder hidden states (last layer)
            decoder_hiddens = outputs.decoder_hidden_states[-1]  # [batch, seq_len, d_model]

            # TOKEN-LEVEL AGE PREDICTION
            # Predict age bracket for each token in sequence
            age_logits_per_token = self.age_predictor(decoder_hiddens)  # [batch, decoder_seq_len, 6]

            # Slice age_logits to match labels sequence length
            # decoder_hiddens may be longer than labels due to teacher forcing
            batch_size, label_seq_len = labels.shape
            age_logits_per_token = age_logits_per_token[:, :label_seq_len, :]  # [batch, label_seq_len, 6]

            # Create mask for diagnosis code tokens only (exclude BOS, <v>, <\v>, padding, etc.)
            # labels: -100 for padding/ignored, >=0 for actual tokens
            # code_offset: token ID where diagnosis codes start (default 6)
            code_mask = (labels >= code_offset) & (labels != -100)  # [batch, label_seq_len]

            # Convert true age to age bracket class
            true_age = x_num[:, 0]  # [batch]
            true_age_class = self.age_to_class(true_age)  # [batch] -> class 0-5

            # Expand true_age_class to match sequence length
            true_age_class_expanded = true_age_class.unsqueeze(1).expand(batch_size, label_seq_len)  # [batch, label_seq_len]

            # Compute cross-entropy loss only for diagnosis code tokens
            if code_mask.any():
                # Flatten to [num_code_tokens, 6] and [num_code_tokens]
                age_logits_codes = age_logits_per_token[code_mask]  # [num_codes, 6]
                age_classes_codes = true_age_class_expanded[code_mask]  # [num_codes]

                age_loss = nn.functional.cross_entropy(age_logits_codes, age_classes_codes)
            else:
                # No code tokens in batch (e.g., all padding) - skip age loss
                age_loss = torch.tensor(0.0, device=decoder_hiddens.device)

            # SEQUENCE-LEVEL SEX PREDICTION (keep as mean pooling - sex is sequence-level property)
            pooled_repr = decoder_hiddens.mean(dim=1)  # [batch, d_model]
            predicted_sex_logits = self.sex_predictor(pooled_repr)  # [batch, 2]
            true_sex = x_cat[:, 0]  # [batch]
            sex_loss = nn.functional.cross_entropy(predicted_sex_logits, true_sex)

            # Store original LM loss
            lm_loss = outputs.loss

            # Combined loss
            total_loss = lm_loss + self.age_loss_weight * age_loss + self.sex_loss_weight * sex_loss

            # Update outputs with detailed losses
            outputs.loss = total_loss
            outputs.age_loss = age_loss
            outputs.sex_loss = sex_loss
            outputs.lm_loss = lm_loss

        return outputs
