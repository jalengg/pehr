 BART Encoder vs Decoder: A Step-by-Step Example

  High-level: BART is an encoder-decoder seq2seq model. The encoder "reads" the input, the decoder "writes" the output.

  Example Patient

  # Patient demographics
  age = 65
  gender = "M"  # (encoded as 0)
  ethnicity = "WHITE"  # (encoded as 0)

  # Patient visit history
  visits = [["V3001", "250.00"], ["401.9"]]

  # Tokenized sequence
  token_ids = [BOS, <v>, V3001_id, 250.00_id, <\v>, <v>, 401.9_id, <\v>, <END>]
  #            1    3    6        7          4      3    8         4      5

  ---
  Step 1: ConditionalPrompt

  Input: Raw demographics
  x_num = torch.tensor([[65.0]])     # [batch=1, n_num_features=1]
  x_cat = torch.tensor([[0, 0]])     # [batch=1, n_cat_features=2] (gender, ethnicity)

  Process: Embed to continuous vectors
  prompt_embeds = ConditionalPrompt(x_num, x_cat)

  # Output shape: [batch=1, n_prompts=3, hidden_dim=768]
  # prompt_embeds[0, 0, :] = age embedding (768 dims)
  # prompt_embeds[0, 1, :] = gender embedding (768 dims)  
  # prompt_embeds[0, 2, :] = ethnicity embedding (768 dims)

  ---
  Step 2: PromptBartEncoder

  Purpose: Encode the input sequence into contextualized representations

  Input:
  input_ids = [1, 3, 6, 7, 4, 3, 8, 4, 5]  # Token IDs for patient sequence
  # Decoded: BOS <v> V3001 250.00 <\v> <v> 401.9 <\v> <END>

  prompt_embeds = [[age_emb], [gender_emb], [ethnicity_emb]]  # [1, 3, 768]

  Process:

  1. Convert tokens to embeddings:
  token_embeds = embed_tokens(input_ids)
  # Shape: [batch=1, seq_len=9, hidden_dim=768]
  # token_embeds[0, 0, :] = embedding for BOS
  # token_embeds[0, 1, :] = embedding for <v>
  # token_embeds[0, 2, :] = embedding for V3001
  # ...

  2. Prepend prompt embeddings:
  inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
  # Shape: [batch=1, 3 + 9 = 12, hidden_dim=768]
  # inputs_embeds[0, 0, :] = age embedding
  # inputs_embeds[0, 1, :] = gender embedding
  # inputs_embeds[0, 2, :] = ethnicity embedding
  # inputs_embeds[0, 3, :] = BOS embedding
  # inputs_embeds[0, 4, :] = <v> embedding
  # inputs_embeds[0, 5, :] = V3001 embedding
  # ...

  3. Add positional encodings:
  positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Position for each token
  pos_embeds = embed_positions(positions)

  hidden_states = inputs_embeds + pos_embeds

  4. Pass through transformer layers:
  # 6 transformer layers (for BART-base)
  for layer in encoder_layers:
      # Self-attention: each token attends to all previous tokens + prompts
      hidden_states = layer(hidden_states)

  # After layer 1:
  # hidden_states[0, 5, :] (V3001) now contains info from:
  #   - age, gender, ethnicity (prompts)
  #   - BOS, <v> (previous tokens)
  #   - itself (V3001)

  # After layer 6:
  # hidden_states[0, 5, :] (V3001) contains info from ENTIRE sequence
  #   including demographics via attention to prompt embeddings

  Output:
  encoder_outputs = {
      'last_hidden_state': hidden_states,  # [batch=1, 12, hidden_dim=768]
      # All 12 tokens (3 prompts + 9 sequence tokens) are now contextualized
  }

  Key insight: Encoder output represents the "meaning" of the full patient history conditioned on demographics.

  ---
  Step 3: PromptBartDecoder

  Purpose: Generate new tokens autoregressively, conditioned on encoder output

  Input (during training with teacher forcing):
  # Same sequence, but shifted right by 1 token
  decoder_input_ids = [BOS, 1, 3, 6, 7, 4, 3, 8, 4]
  # Original:         [BOS, <v>, V3001, 250.00, <\v>, <v>, 401.9, <\v>, <END>]
  # Decoder input:    [BOS, BOS, <v>, V3001, 250.00, <\v>, <v>, 401.9, <\v>]
  #                    ↑ Start token, then predict next token from previous

  encoder_outputs = hidden_states from encoder  # [1, 12, 768]
  prompt_embeds = same as encoder  # [1, 3, 768]

  Process:

  1. Convert decoder input to embeddings:
  decoder_token_embeds = embed_tokens(decoder_input_ids)
  # Shape: [batch=1, tgt_len=9, hidden_dim=768]

  2. Prepend prompt embeddings:
  decoder_inputs_embeds = torch.cat([prompt_embeds, decoder_token_embeds], dim=1)
  # Shape: [batch=1, 3 + 9 = 12, hidden_dim=768]

  3. Pass through decoder layers:
  for layer in decoder_layers:
      # a. Self-attention: each decoder token attends to previous decoder tokens + prompts
      #    (causal mask prevents attending to future tokens)

      # b. Cross-attention: attend to encoder outputs
      #    Decoder queries encoder for relevant context

      hidden_states = layer(
          hidden_states,
          encoder_hidden_states=encoder_outputs  # Cross-attention
      )

  Cross-Attention Example:

  When generating token at position 5 (predicting <\v> after 250.00):

  # Self-attention sees:
  # [age_emb, gender_emb, ethnicity_emb, BOS, BOS, <v>, V3001, 250.00]
  # (can't see future: <\v>, <v>, 401.9, etc.)

  # Cross-attention queries encoder:
  query = hidden_states[0, 7, :]  # Current position (250.00)
  keys = encoder_outputs[:, :, :]  # All encoder outputs (prompts + full sequence)

  # Attention weights might look like:
  attention_weights = [
      0.05,  # age (prompt)
      0.02,  # gender (prompt)
      0.02,  # ethnicity (prompt)
      0.01,  # BOS
      0.10,  # <v> (visit start)
      0.30,  # V3001 (previous code - highly relevant)
      0.45,  # 250.00 (current code)
      0.03,  # <\v>
      0.01,  # <v>
      0.01,  # 401.9
      ...
  ]

  # Decoder learns: after V3001 and 250.00, likely generate <\v> (visit end)

  Output:
  decoder_outputs = {
      'last_hidden_state': hidden_states,  # [batch=1, 12, hidden_dim=768]
  }

  4. Language modeling head:
  logits = lm_head(hidden_states)  # [batch=1, 12, vocab_size]

  # For each position, logits give probability distribution over vocabulary
  # logits[0, 7, :] = probabilities for next token after 250.00
  # logits[0, 7, V_END_TOKEN_ID] = high (likely to predict <\v>)

  ---
  Data Flow Summary

  Demographics (x_num, x_cat)
      ↓
  ConditionalPrompt
      ↓
  prompt_embeds [1, 3, 768]  ← 3 demographic embeddings
      ↓
      ├─→ PromptBartEncoder
      │       ↓
      │   Prepend to input_ids: [age, gender, ethnicity, BOS, <v>, V3001, ...]
      │       ↓
      │   6 Transformer layers (self-attention)
      │       ↓
      │   encoder_outputs [1, 12, 768]  ← Contextualized representations
      │       ↓
      └─→ PromptBartDecoder
              ↓
          Prepend to decoder_input_ids: [age, gender, ethnicity, BOS, BOS, <v>, ...]
              ↓
          6 Transformer layers:
            - Self-attention (causal)
            - Cross-attention to encoder_outputs
              ↓
          decoder_outputs [1, 12, 768]
              ↓
          lm_head (linear projection)
              ↓
          logits [1, 12, vocab_size]  ← Probabilities for next token at each position
              ↓
          Loss = CrossEntropy(logits, labels)

  ---
  Generation (Inference)

  During generation, decoder is called autoregressively:

  # Step 1: Generate first token
  decoder_input = [BOS]
  prompt_embeds = ConditionalPrompt(age=65, gender=M, ethnicity=WHITE)

  encoder_outputs = PromptBartEncoder(input_ids, prompt_embeds)

  decoder_outputs = PromptBartDecoder(
      [BOS],  # Only start token
      encoder_outputs,
      prompt_embeds
  )

  logits = lm_head(decoder_outputs)
  next_token = sample(logits[:, -1, :])  # Sample from last position
  # next_token = <v> (visit start)

  # Step 2: Generate second token
  decoder_input = [BOS, <v>]
  decoder_outputs = PromptBartDecoder(
      [BOS, <v>],
      encoder_outputs,
      prompt_embeds
  )
  next_token = sample(logits[:, -1, :])
  # next_token = V3001 (first diagnosis code)

  # Continue until <END> token...
