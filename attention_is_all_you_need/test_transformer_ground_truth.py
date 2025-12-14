"""
Tests for: Attention Is All You Need
Paper: https://arxiv.org/abs/1706.03762 (Vaswani et al., 2017)

Instructions:
- Each test has a `# TODO: IMPLEMENT` section
- Replace `None` or `pass` with your implementation
- Run with: pytest test_transformer.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# SECTION 1: Scaled Dot-Product Attention
# Reference: Section 3.2.1, Equation 1
# ============================================================================

class TestScaledDotProductAttention:
    """
    Scaled Dot-Product Attention computes attention weights from queries and keys,
    then applies them to values. The scaling factor 1/sqrt(d_k) prevents the dot
    products from growing large in magnitude for high-dimensional keys.

    Key equation (Equation 1):
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Where:
        - Q: queries of shape (batch, seq_len_q, d_k)
        - K: keys of shape (batch, seq_len_k, d_k)
        - V: values of shape (batch, seq_len_k, d_v)
        - Output: shape (batch, seq_len_q, d_v)
    """

    def test_attention_output_shape(self):
        """Verify output shape matches (batch, seq_len_q, d_v)."""
        batch_size, seq_len_q, seq_len_k, d_k, d_v = 2, 10, 15, 64, 64

        Q = torch.randn(batch_size, seq_len_q, d_k)
        K = torch.randn(batch_size, seq_len_k, d_k)
        V = torch.randn(batch_size, seq_len_k, d_v)

        # TODO: IMPLEMENT
        # Implement scaled dot-product attention from Equation 1
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        output = step_4

        # Verification
        assert output is not None, "Implementation required"
        expected_shape = (batch_size, seq_len_q, d_v)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_attention_weights_sum_to_one(self):
        """Verify attention weights (after softmax) sum to 1 along key dimension."""
        batch_size, seq_len_q, seq_len_k, d_k = 2, 5, 8, 32

        Q = torch.randn(batch_size, seq_len_q, d_k)
        K = torch.randn(batch_size, seq_len_k, d_k)

        # TODO: IMPLEMENT
        # Compute attention weights (before multiplying by V)
        # scores = QK^T / sqrt(d_k), then softmax
        attention_weights = None  # <-- YOUR CODE HERE (shape: batch, seq_len_q, seq_len_k)

        # Verification
        assert attention_weights is not None, "Implementation required"
        assert attention_weights.shape == (batch_size, seq_len_q, seq_len_k)

        # Softmax should sum to 1 along the last dimension (keys)
        weight_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
            "Attention weights should sum to 1 along key dimension"

    def test_scaling_factor(self):
        """Verify the scaling by 1/sqrt(d_k) is applied correctly."""
        batch_size, seq_len, d_k = 1, 4, 64

        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)

        # Compute raw scores (without scaling)
        raw_scores = torch.bmm(Q, K.transpose(-2, -1))

        # TODO: IMPLEMENT
        # Compute scaled scores (with 1/sqrt(d_k) scaling)
        scaled_scores = None  # <-- YOUR CODE HERE

        # Verification
        assert scaled_scores is not None, "Implementation required"
        expected_scaled = raw_scores / math.sqrt(d_k)
        assert torch.allclose(scaled_scores, expected_scaled, atol=1e-5), \
            "Scores should be scaled by 1/sqrt(d_k)"

    def test_attention_with_mask(self):
        """Verify masking sets attention weights to ~0 for masked positions."""
        batch_size, seq_len, d_k, d_v = 2, 5, 32, 32

        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_v)

        # Create a mask: True means "mask out" (set to -inf before softmax)
        # Mask out the last 2 positions in the key sequence
        mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, -2:] = True

        # TODO: IMPLEMENT
        # Implement scaled dot-product attention with masking
        # Where mask is True, set scores to -inf (or very large negative) before softmax
        output, attention_weights = None, None  # <-- YOUR CODE HERE

        # Verification
        assert output is not None and attention_weights is not None, "Implementation required"

        # Masked positions should have near-zero attention weights
        masked_weights = attention_weights[:, :, -2:]
        assert (masked_weights < 1e-6).all(), \
            "Masked positions should have near-zero attention weights"

    def test_gradient_flow(self):
        """Verify gradients flow back through attention."""
        batch_size, seq_len, d_k, d_v = 2, 6, 32, 32

        Q = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_v, requires_grad=True)

        # TODO: IMPLEMENT
        output = None  # <-- YOUR CODE HERE

        # Verification
        assert output is not None, "Implementation required"
        loss = output.sum()
        loss.backward()

        assert Q.grad is not None, "Gradients should flow to Q"
        assert K.grad is not None, "Gradients should flow to K"
        assert V.grad is not None, "Gradients should flow to V"
        assert not torch.isnan(Q.grad).any(), "Gradients should not be NaN"


# ============================================================================
# SECTION 2: Multi-Head Attention
# Reference: Section 3.2.2
# ============================================================================

class TestMultiHeadAttention:
    """
    Multi-Head Attention allows the model to jointly attend to information from
    different representation subspaces at different positions.

    Key equations:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    Paper uses:
        - h = 8 heads
        - d_k = d_v = d_model / h = 64
        - d_model = 512
    """

    def test_multihead_output_shape(self):
        """Verify MultiHead output has shape (batch, seq_len, d_model)."""
        batch_size, seq_len, d_model, n_heads = 2, 10, 512, 8

        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)

        # TODO: IMPLEMENT
        # Create a MultiHeadAttention module and compute output
        # The module should have learnable projection matrices W^Q, W^K, W^V, W^O
        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, Q, K, V, mask=None):
                # <-- YOUR CODE HERE
                pass

        mha = MultiHeadAttention(d_model, n_heads)
        output = mha(Q, K, V)

        # Verification
        assert output is not None, "Implementation required"
        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_multihead_different_qkv_lengths(self):
        """Verify MultiHead works with different query and key/value sequence lengths."""
        batch_size, seq_len_q, seq_len_kv, d_model, n_heads = 2, 10, 15, 512, 8

        Q = torch.randn(batch_size, seq_len_q, d_model)
        K = torch.randn(batch_size, seq_len_kv, d_model)
        V = torch.randn(batch_size, seq_len_kv, d_model)

        # TODO: IMPLEMENT
        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, Q, K, V, mask=None):
                # <-- YOUR CODE HERE
                pass

        mha = MultiHeadAttention(d_model, n_heads)
        output = mha(Q, K, V)

        # Verification
        assert output is not None, "Implementation required"
        expected_shape = (batch_size, seq_len_q, d_model)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_multihead_parameter_count(self):
        """Verify the number of learnable parameters matches expected count."""
        d_model, n_heads = 512, 8
        d_k = d_v = d_model // n_heads  # 64

        # TODO: IMPLEMENT
        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, Q, K, V, mask=None):
                # <-- YOUR CODE HERE
                pass

        mha = MultiHeadAttention(d_model, n_heads)

        # Expected parameters (without biases):
        # W^Q: d_model x d_model = 512 x 512
        # W^K: d_model x d_model = 512 x 512
        # W^V: d_model x d_model = 512 x 512
        # W^O: d_model x d_model = 512 x 512
        # Total: 4 * 512 * 512 = 1,048,576

        total_params = sum(p.numel() for p in mha.parameters())
        expected_params = 4 * d_model * d_model

        # Verification (allow for bias terms if included)
        assert total_params >= expected_params, \
            f"Expected at least {expected_params} parameters, got {total_params}"

    def test_multihead_gradient_flow(self):
        """Verify gradients flow through all projection matrices."""
        batch_size, seq_len, d_model, n_heads = 2, 8, 256, 4

        Q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        # TODO: IMPLEMENT
        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, Q, K, V, mask=None):
                # <-- YOUR CODE HERE
                pass

        mha = MultiHeadAttention(d_model, n_heads)
        output = mha(Q, K, V)

        # Verification
        assert output is not None, "Implementation required"
        loss = output.sum()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in mha.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# ============================================================================
# SECTION 3: Positional Encoding
# Reference: Section 3.5
# ============================================================================

class TestPositionalEncoding:
    """
    Since the Transformer contains no recurrence or convolution, positional
    encodings are added to give the model information about token positions.

    Key equations:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
        - pos is the position (0 to max_seq_len-1)
        - i is the dimension index (0 to d_model/2 - 1)
        - The encoding has the same dimension d_model as embeddings so they can be summed
    """

    def test_positional_encoding_shape(self):
        """Verify PE has shape (max_seq_len, d_model)."""
        max_seq_len, d_model = 100, 512

        # TODO: IMPLEMENT
        # Generate positional encodings for all positions
        pe = None  # <-- YOUR CODE HERE (shape: max_seq_len, d_model)

        # Verification
        assert pe is not None, "Implementation required"
        expected_shape = (max_seq_len, d_model)
        assert pe.shape == expected_shape, f"Expected {expected_shape}, got {pe.shape}"

    def test_positional_encoding_sin_cos_pattern(self):
        """Verify even indices use sin, odd indices use cos."""
        max_seq_len, d_model = 50, 16

        # TODO: IMPLEMENT
        pe = None  # <-- YOUR CODE HERE

        # Verification
        assert pe is not None, "Implementation required"

        # Manually compute expected values for position 0
        pos = 0
        for i in range(d_model // 2):
            div_term = 10000 ** (2 * i / d_model)
            expected_sin = math.sin(pos / div_term)
            expected_cos = math.cos(pos / div_term)

            assert torch.isclose(pe[pos, 2*i], torch.tensor(expected_sin), atol=1e-5), \
                f"PE[{pos}, {2*i}] should be sin"
            assert torch.isclose(pe[pos, 2*i+1], torch.tensor(expected_cos), atol=1e-5), \
                f"PE[{pos}, {2*i+1}] should be cos"

    def test_positional_encoding_bounded(self):
        """Verify PE values are bounded in [-1, 1] (sin/cos range)."""
        max_seq_len, d_model = 1000, 512

        # TODO: IMPLEMENT
        pe = None  # <-- YOUR CODE HERE

        # Verification
        assert pe is not None, "Implementation required"
        assert (pe >= -1.0).all() and (pe <= 1.0).all(), \
            "Positional encoding values should be in [-1, 1]"

    def test_positional_encoding_different_positions(self):
        """Verify different positions have different encodings."""
        max_seq_len, d_model = 100, 256

        # TODO: IMPLEMENT
        pe = None  # <-- YOUR CODE HERE

        # Verification
        assert pe is not None, "Implementation required"

        # Check that consecutive positions have different encodings
        for pos in range(min(10, max_seq_len - 1)):
            assert not torch.allclose(pe[pos], pe[pos + 1]), \
                f"Positions {pos} and {pos + 1} should have different encodings"

    def test_positional_encoding_deterministic(self):
        """Verify PE is deterministic (same inputs give same outputs)."""
        max_seq_len, d_model = 50, 128

        # TODO: IMPLEMENT
        # Generate PE twice
        pe1 = None  # <-- YOUR CODE HERE
        pe2 = None  # <-- YOUR CODE HERE

        # Verification
        assert pe1 is not None and pe2 is not None, "Implementation required"
        assert torch.allclose(pe1, pe2), "Positional encoding should be deterministic"


# ============================================================================
# SECTION 4: Position-wise Feed-Forward Network
# Reference: Section 3.3, Equation 2
# ============================================================================

class TestPositionwiseFeedForward:
    """
    Each encoder/decoder layer contains a fully connected feed-forward network
    applied to each position separately and identically.

    Key equation (Equation 2):
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    This is two linear transformations with a ReLU activation in between.
    Paper uses:
        - d_model = 512 (input/output dimension)
        - d_ff = 2048 (inner-layer dimension)
    """

    def test_ffn_output_shape(self):
        """Verify FFN preserves input shape (batch, seq_len, d_model)."""
        batch_size, seq_len, d_model, d_ff = 2, 10, 512, 2048

        x = torch.randn(batch_size, seq_len, d_model)

        # TODO: IMPLEMENT
        # Create a position-wise feed-forward network
        class PositionwiseFeedForward(nn.Module):
            def __init__(self, d_model, d_ff):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, x):
                # FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
                # <-- YOUR CODE HERE
                pass

        ffn = PositionwiseFeedForward(d_model, d_ff)
        output = ffn(x)

        # Verification
        assert output is not None, "Implementation required"
        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_ffn_uses_relu(self):
        """Verify ReLU is applied (negative values in hidden layer become 0)."""
        d_model, d_ff = 16, 32

        # Create input that will produce negative hidden values
        x = torch.randn(1, 1, d_model)

        # TODO: IMPLEMENT
        class PositionwiseFeedForward(nn.Module):
            def __init__(self, d_model, d_ff):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, x):
                # <-- YOUR CODE HERE
                pass

            def get_hidden(self, x):
                """Return hidden representation before ReLU for testing."""
                # <-- YOUR CODE HERE (return xW_1 + b_1)
                pass

        ffn = PositionwiseFeedForward(d_model, d_ff)

        # Verification: hidden after ReLU should have no negative values
        # (Though this is implicitly tested by the correct output shape and gradient flow)
        output = ffn(x)
        assert output is not None, "Implementation required"

    def test_ffn_parameter_count(self):
        """Verify parameter count matches expected."""
        d_model, d_ff = 512, 2048

        # TODO: IMPLEMENT
        class PositionwiseFeedForward(nn.Module):
            def __init__(self, d_model, d_ff):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, x):
                # <-- YOUR CODE HERE
                pass

        ffn = PositionwiseFeedForward(d_model, d_ff)

        # Expected parameters:
        # W_1: d_model x d_ff + b_1: d_ff = 512*2048 + 2048
        # W_2: d_ff x d_model + b_2: d_model = 2048*512 + 512
        expected_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
        total_params = sum(p.numel() for p in ffn.parameters())

        # Verification
        assert total_params == expected_params, \
            f"Expected {expected_params} parameters, got {total_params}"

    def test_ffn_gradient_flow(self):
        """Verify gradients flow through FFN."""
        batch_size, seq_len, d_model, d_ff = 2, 5, 64, 128

        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        # TODO: IMPLEMENT
        class PositionwiseFeedForward(nn.Module):
            def __init__(self, d_model, d_ff):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, x):
                # <-- YOUR CODE HERE
                pass

        ffn = PositionwiseFeedForward(d_model, d_ff)
        output = ffn(x)

        # Verification
        assert output is not None, "Implementation required"
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        for name, param in ffn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ============================================================================
# SECTION 5: Layer Normalization
# Reference: Section 3.1 (uses Layer Normalization from Ba et al., 2016)
# ============================================================================

class TestLayerNormalization:
    """
    The Transformer uses layer normalization after each sub-layer.
    The output of each sub-layer is: LayerNorm(x + Sublayer(x))

    Layer normalization normalizes across the feature dimension, computing
    mean and variance for each position independently.
    """

    def test_layer_norm_output_shape(self):
        """Verify LayerNorm preserves input shape."""
        batch_size, seq_len, d_model = 2, 10, 512

        x = torch.randn(batch_size, seq_len, d_model)

        # TODO: IMPLEMENT
        # Apply layer normalization (can use nn.LayerNorm or implement from scratch)
        output = None  # <-- YOUR CODE HERE

        # Verification
        assert output is not None, "Implementation required"
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_layer_norm_normalized_stats(self):
        """Verify LayerNorm produces mean ~0 and var ~1 across features."""
        batch_size, seq_len, d_model = 4, 8, 256

        x = torch.randn(batch_size, seq_len, d_model) * 10 + 5  # Not normalized

        # TODO: IMPLEMENT
        output = None  # <-- YOUR CODE HERE

        # Verification
        assert output is not None, "Implementation required"

        # Mean should be ~0 and variance ~1 across the last dimension
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), \
            "Mean across features should be ~0"
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4), \
            "Variance across features should be ~1"


# ============================================================================
# SECTION 6: Encoder Layer
# Reference: Section 3.1
# ============================================================================

class TestEncoderLayer:
    """
    Each encoder layer has two sub-layers:
    1. Multi-head self-attention
    2. Position-wise feed-forward network

    Each sub-layer has a residual connection followed by layer normalization:
        output = LayerNorm(x + Sublayer(x))
    """

    def test_encoder_layer_output_shape(self):
        """Verify encoder layer preserves input shape."""
        batch_size, seq_len, d_model, n_heads, d_ff = 2, 10, 512, 8, 2048

        x = torch.randn(batch_size, seq_len, d_model)

        # TODO: IMPLEMENT
        class EncoderLayer(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()
                # <-- YOUR CODE HERE
                # Include: multi-head attention, feed-forward, layer norms, dropout
                pass

            def forward(self, x, mask=None):
                # <-- YOUR CODE HERE
                # 1. Self-attention with residual + layer norm
                # 2. Feed-forward with residual + layer norm
                pass

        encoder_layer = EncoderLayer(d_model, n_heads, d_ff)
        output = encoder_layer(x)

        # Verification
        assert output is not None, "Implementation required"
        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_encoder_layer_residual_connection(self):
        """Verify residual connections are present (output changes when input changes)."""
        batch_size, seq_len, d_model, n_heads, d_ff = 1, 5, 64, 4, 128

        # TODO: IMPLEMENT
        class EncoderLayer(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, x, mask=None):
                # <-- YOUR CODE HERE
                pass

        encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout=0.0)
        encoder_layer.eval()

        x1 = torch.randn(batch_size, seq_len, d_model)
        x2 = x1.clone()
        x2[:, 0, :] += 1.0  # Modify first position

        output1 = encoder_layer(x1)
        output2 = encoder_layer(x2)

        # Verification
        assert output1 is not None and output2 is not None, "Implementation required"
        # Outputs should differ where input differs
        assert not torch.allclose(output1, output2), \
            "Output should change when input changes (residual connection)"

    def test_encoder_layer_gradient_flow(self):
        """Verify gradients flow through encoder layer."""
        batch_size, seq_len, d_model, n_heads, d_ff = 2, 8, 128, 4, 256

        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        # TODO: IMPLEMENT
        class EncoderLayer(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, x, mask=None):
                # <-- YOUR CODE HERE
                pass

        encoder_layer = EncoderLayer(d_model, n_heads, d_ff)
        output = encoder_layer(x)

        # Verification
        assert output is not None, "Implementation required"
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(x.grad).any(), "Gradients should not be NaN"


# ============================================================================
# SECTION 7: Decoder Layer with Causal Masking
# Reference: Section 3.1
# ============================================================================

class TestDecoderLayer:
    """
    Each decoder layer has three sub-layers:
    1. Masked multi-head self-attention (causal mask prevents attending to future)
    2. Multi-head encoder-decoder attention (queries from decoder, K/V from encoder)
    3. Position-wise feed-forward network

    The masking ensures predictions for position i depend only on known outputs
    at positions less than i.
    """

    def test_causal_mask_shape(self):
        """Verify causal mask has correct shape and pattern."""
        seq_len = 5

        # TODO: IMPLEMENT
        # Create a causal mask: positions can only attend to previous positions
        # mask[i, j] should be True (masked) if j > i
        causal_mask = None  # <-- YOUR CODE HERE (shape: seq_len, seq_len)

        # Verification
        assert causal_mask is not None, "Implementation required"
        assert causal_mask.shape == (seq_len, seq_len), \
            f"Expected shape ({seq_len}, {seq_len})"

        # Check pattern: lower triangular should be False (not masked)
        # Upper triangular should be True (masked)
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert causal_mask[i, j], f"Position ({i}, {j}) should be masked"
                else:
                    assert not causal_mask[i, j], f"Position ({i}, {j}) should not be masked"

    def test_decoder_layer_output_shape(self):
        """Verify decoder layer output shape."""
        batch_size, tgt_len, src_len, d_model, n_heads, d_ff = 2, 8, 12, 512, 8, 2048

        tgt = torch.randn(batch_size, tgt_len, d_model)  # Decoder input
        memory = torch.randn(batch_size, src_len, d_model)  # Encoder output

        # TODO: IMPLEMENT
        class DecoderLayer(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()
                # <-- YOUR CODE HERE
                # Include: masked self-attention, encoder-decoder attention,
                #          feed-forward, layer norms, dropout
                pass

            def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
                # <-- YOUR CODE HERE
                # 1. Masked self-attention on tgt
                # 2. Encoder-decoder attention (Q from tgt, K/V from memory)
                # 3. Feed-forward
                pass

        decoder_layer = DecoderLayer(d_model, n_heads, d_ff)
        output = decoder_layer(tgt, memory)

        # Verification
        assert output is not None, "Implementation required"
        expected_shape = (batch_size, tgt_len, d_model)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_causal_attention_no_future_leakage(self):
        """Verify causal attention prevents attending to future positions."""
        batch_size, seq_len, d_k, d_v = 1, 4, 32, 32

        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_v)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # TODO: IMPLEMENT
        # Compute masked attention and return attention weights
        attention_weights = None  # <-- YOUR CODE HERE (shape: batch, seq_len, seq_len)

        # Verification
        assert attention_weights is not None, "Implementation required"

        # Upper triangular (future positions) should have ~0 attention
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert attention_weights[0, i, j] < 1e-6, \
                    f"Position {i} should not attend to future position {j}"


# ============================================================================
# SECTION 8: Full Encoder and Decoder Stacks
# Reference: Section 3.1
# ============================================================================

class TestEncoderDecoder:
    """
    The full Transformer encoder consists of N=6 identical layers.
    The full Transformer decoder also consists of N=6 identical layers.
    """

    def test_encoder_stack_output_shape(self):
        """Verify stacked encoder output shape."""
        batch_size, seq_len, d_model, n_heads, d_ff, n_layers = 2, 10, 512, 8, 2048, 6

        x = torch.randn(batch_size, seq_len, d_model)

        # TODO: IMPLEMENT
        class Encoder(nn.Module):
            def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()
                # <-- YOUR CODE HERE
                # Stack of N encoder layers
                pass

            def forward(self, x, mask=None):
                # <-- YOUR CODE HERE
                pass

        encoder = Encoder(n_layers, d_model, n_heads, d_ff)
        output = encoder(x)

        # Verification
        assert output is not None, "Implementation required"
        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_decoder_stack_output_shape(self):
        """Verify stacked decoder output shape."""
        batch_size, tgt_len, src_len, d_model, n_heads, d_ff, n_layers = 2, 8, 12, 512, 8, 2048, 6

        tgt = torch.randn(batch_size, tgt_len, d_model)
        memory = torch.randn(batch_size, src_len, d_model)

        # TODO: IMPLEMENT
        class Decoder(nn.Module):
            def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()
                # <-- YOUR CODE HERE
                # Stack of N decoder layers
                pass

            def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
                # <-- YOUR CODE HERE
                pass

        decoder = Decoder(n_layers, d_model, n_heads, d_ff)
        output = decoder(tgt, memory)

        # Verification
        assert output is not None, "Implementation required"
        expected_shape = (batch_size, tgt_len, d_model)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


# ============================================================================
# SECTION 9: Learning Rate Schedule
# Reference: Section 5.3, Equation 3
# ============================================================================

class TestLearningRateSchedule:
    """
    The paper uses a custom learning rate schedule with warmup.

    Key equation (Equation 3):
        lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    This increases linearly for the first warmup_steps, then decreases proportionally
    to the inverse square root of the step number.
    """

    def test_lr_warmup_linear_increase(self):
        """Verify LR increases linearly during warmup."""
        d_model = 512
        warmup_steps = 4000

        # TODO: IMPLEMENT
        def get_lr(step, d_model, warmup_steps):
            # Implement Equation 3
            # <-- YOUR CODE HERE
            pass

        # Check that LR increases during warmup
        lr_1000 = get_lr(1000, d_model, warmup_steps)
        lr_2000 = get_lr(2000, d_model, warmup_steps)
        lr_3000 = get_lr(3000, d_model, warmup_steps)

        # Verification
        assert lr_1000 is not None, "Implementation required"
        assert lr_2000 > lr_1000, "LR should increase during warmup"
        assert lr_3000 > lr_2000, "LR should increase during warmup"

    def test_lr_peak_at_warmup_steps(self):
        """Verify LR peaks around warmup_steps."""
        d_model = 512
        warmup_steps = 4000

        # TODO: IMPLEMENT
        def get_lr(step, d_model, warmup_steps):
            # <-- YOUR CODE HERE
            pass

        # Check LR around warmup_steps
        lr_before = get_lr(warmup_steps - 100, d_model, warmup_steps)
        lr_at = get_lr(warmup_steps, d_model, warmup_steps)
        lr_after = get_lr(warmup_steps + 100, d_model, warmup_steps)

        # Verification
        assert lr_at is not None, "Implementation required"
        # LR at warmup_steps should be >= nearby values
        assert lr_at >= lr_before * 0.99, "LR should peak around warmup_steps"
        assert lr_at >= lr_after * 0.99, "LR should peak around warmup_steps"

    def test_lr_decay_after_warmup(self):
        """Verify LR decays after warmup."""
        d_model = 512
        warmup_steps = 4000

        # TODO: IMPLEMENT
        def get_lr(step, d_model, warmup_steps):
            # <-- YOUR CODE HERE
            pass

        # Check that LR decreases after warmup
        lr_5000 = get_lr(5000, d_model, warmup_steps)
        lr_10000 = get_lr(10000, d_model, warmup_steps)
        lr_50000 = get_lr(50000, d_model, warmup_steps)

        # Verification
        assert lr_5000 is not None, "Implementation required"
        assert lr_10000 < lr_5000, "LR should decrease after warmup"
        assert lr_50000 < lr_10000, "LR should continue decreasing"

    def test_lr_scales_with_d_model(self):
        """Verify LR scales with d_model^(-0.5)."""
        warmup_steps = 4000
        step = 10000

        # TODO: IMPLEMENT
        def get_lr(step, d_model, warmup_steps):
            # <-- YOUR CODE HERE
            pass

        lr_256 = get_lr(step, 256, warmup_steps)
        lr_512 = get_lr(step, 512, warmup_steps)
        lr_1024 = get_lr(step, 1024, warmup_steps)

        # Verification
        assert lr_256 is not None, "Implementation required"

        # LR ratio should match sqrt(d_model) ratio
        expected_ratio = math.sqrt(512 / 256)
        actual_ratio = lr_256 / lr_512
        assert abs(actual_ratio - expected_ratio) < 0.01, \
            f"LR should scale with d_model^(-0.5), expected ratio {expected_ratio}, got {actual_ratio}"


# ============================================================================
# SECTION 10: Label Smoothing
# Reference: Section 5.4
# ============================================================================

class TestLabelSmoothing:
    """
    The paper uses label smoothing with epsilon_ls = 0.1.
    Instead of using hard targets (one-hot), the target distribution is smoothed.

    For a correct class, the target is (1 - epsilon) instead of 1.
    The remaining epsilon is distributed among other classes.
    """

    def test_label_smoothing_distribution(self):
        """Verify label smoothing produces correct target distribution."""
        num_classes = 10
        epsilon = 0.1
        target_class = 3

        # TODO: IMPLEMENT
        # Create smoothed label distribution for a single example
        # Correct class gets (1 - epsilon), others get epsilon / (num_classes - 1)
        smoothed_target = None  # <-- YOUR CODE HERE (shape: num_classes)

        # Verification
        assert smoothed_target is not None, "Implementation required"
        assert smoothed_target.shape == (num_classes,), \
            f"Expected shape ({num_classes},)"

        # Check target class has (1 - epsilon)
        expected_target = 1 - epsilon
        assert torch.isclose(smoothed_target[target_class],
                            torch.tensor(expected_target), atol=1e-5), \
            f"Target class should have probability {expected_target}"

        # Check other classes have epsilon / (num_classes - 1)
        expected_other = epsilon / (num_classes - 1)
        for i in range(num_classes):
            if i != target_class:
                assert torch.isclose(smoothed_target[i],
                                    torch.tensor(expected_other), atol=1e-5), \
                    f"Other classes should have probability {expected_other}"

    def test_label_smoothing_sums_to_one(self):
        """Verify smoothed labels sum to 1."""
        num_classes = 100
        epsilon = 0.1
        batch_size = 8
        targets = torch.randint(0, num_classes, (batch_size,))

        # TODO: IMPLEMENT
        # Create smoothed labels for a batch
        smoothed_targets = None  # <-- YOUR CODE HERE (shape: batch_size, num_classes)

        # Verification
        assert smoothed_targets is not None, "Implementation required"

        sums = smoothed_targets.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), \
            "Smoothed labels should sum to 1"

    def test_label_smoothing_loss(self):
        """Verify label smoothing can be used in cross-entropy loss."""
        batch_size, num_classes = 4, 10
        epsilon = 0.1

        logits = torch.randn(batch_size, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,))

        # TODO: IMPLEMENT
        # Compute cross-entropy loss with label smoothing
        # This is KL divergence between softmax(logits) and smoothed targets
        loss = None  # <-- YOUR CODE HERE

        # Verification
        assert loss is not None, "Implementation required"
        assert loss.ndim == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

        # Verify gradient flow
        loss.backward()
        assert logits.grad is not None, "Gradients should flow to logits"


# ============================================================================
# SECTION 11: Complete Transformer Model
# Reference: Full architecture from Figure 1
# ============================================================================

class TestTransformer:
    """
    The complete Transformer for sequence-to-sequence tasks:
    1. Source embedding + positional encoding
    2. Encoder stack
    3. Target embedding + positional encoding
    4. Decoder stack
    5. Linear layer + softmax for output probabilities

    The paper shares weights between embedding layers and pre-softmax linear layer,
    multiplying embeddings by sqrt(d_model).
    """

    def test_transformer_output_shape(self):
        """Verify Transformer output has correct vocabulary distribution shape."""
        batch_size, src_len, tgt_len = 2, 10, 8
        d_model, n_heads, d_ff, n_layers = 512, 8, 2048, 6
        src_vocab_size, tgt_vocab_size = 10000, 10000

        src = torch.randint(0, src_vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

        # TODO: IMPLEMENT
        class Transformer(nn.Module):
            def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads,
                         d_ff, n_layers, dropout=0.1):
                super().__init__()
                # <-- YOUR CODE HERE
                # Include: embeddings, positional encoding, encoder, decoder, output projection
                pass

            def forward(self, src, tgt, src_mask=None, tgt_mask=None):
                # <-- YOUR CODE HERE
                pass

        transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads,
                                  d_ff, n_layers)
        output = transformer(src, tgt)

        # Verification
        assert output is not None, "Implementation required"
        expected_shape = (batch_size, tgt_len, tgt_vocab_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_embedding_scaling(self):
        """Verify embeddings are scaled by sqrt(d_model)."""
        vocab_size, d_model = 1000, 512

        # TODO: IMPLEMENT
        # Create embedding layer and verify scaling
        class ScaledEmbedding(nn.Module):
            def __init__(self, vocab_size, d_model):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, x):
                # Return embedding * sqrt(d_model)
                # <-- YOUR CODE HERE
                pass

        embedding = ScaledEmbedding(vocab_size, d_model)
        x = torch.tensor([[0, 1, 2]])
        output = embedding(x)

        # Verification
        assert output is not None, "Implementation required"

        # Check scaling: embedding variance should be ~d_model times larger
        # than unscaled embedding variance
        raw_embedding = embedding.embedding.weight[0:3]  # Assuming attribute name
        scaled_output = output[0]

        # The ratio of norms should be approximately sqrt(d_model)
        scale_factor = math.sqrt(d_model)
        ratio = scaled_output.norm() / raw_embedding.norm()
        assert abs(ratio - scale_factor) < 0.1, \
            f"Embedding should be scaled by sqrt(d_model)={scale_factor}, got ratio {ratio}"

    def test_transformer_gradient_flow(self):
        """Verify gradients flow through entire Transformer."""
        batch_size, src_len, tgt_len = 2, 6, 4
        d_model, n_heads, d_ff, n_layers = 64, 4, 128, 2
        src_vocab_size, tgt_vocab_size = 100, 100

        src = torch.randint(0, src_vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

        # TODO: IMPLEMENT
        class Transformer(nn.Module):
            def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads,
                         d_ff, n_layers, dropout=0.1):
                super().__init__()
                # <-- YOUR CODE HERE
                pass

            def forward(self, src, tgt, src_mask=None, tgt_mask=None):
                # <-- YOUR CODE HERE
                pass

        transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads,
                                  d_ff, n_layers)
        output = transformer(src, tgt)

        # Verification
        assert output is not None, "Implementation required"
        loss = output.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in transformer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# ============================================================================
# SECTION 12: Numerical Stability Tests
# Reference: General best practices
# ============================================================================

class TestNumericalStability:
    """
    Tests to ensure the implementation is numerically stable.
    """

    def test_attention_with_large_values(self):
        """Verify attention handles large input values without overflow."""
        batch_size, seq_len, d_k, d_v = 2, 10, 64, 64

        # Large values that could cause overflow without proper scaling
        Q = torch.randn(batch_size, seq_len, d_k) * 100
        K = torch.randn(batch_size, seq_len, d_k) * 100
        V = torch.randn(batch_size, seq_len, d_v)

        # TODO: IMPLEMENT
        output = None  # <-- YOUR CODE HERE

        # Verification
        assert output is not None, "Implementation required"
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isinf(output).any(), "Output should not contain Inf"

    def test_softmax_numerical_stability(self):
        """Verify softmax is stable for extreme values."""
        batch_size, seq_len = 2, 10

        # Extreme values that could cause overflow in naive softmax
        scores = torch.tensor([[1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
                               [-1000.0, -999.0, -998.0, -997.0, -996.0]])

        # TODO: IMPLEMENT
        # Compute stable softmax (subtract max before exp)
        attention_weights = None  # <-- YOUR CODE HERE

        # Verification
        assert attention_weights is not None, "Implementation required"
        assert not torch.isnan(attention_weights).any(), "Weights should not be NaN"
        assert not torch.isinf(attention_weights).any(), "Weights should not be Inf"
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(2), atol=1e-5), \
            "Softmax should sum to 1"

    def test_layer_norm_with_constant_input(self):
        """Verify layer norm handles constant input (zero variance) gracefully."""
        batch_size, seq_len, d_model = 2, 5, 64

        # Constant input has zero variance
        x = torch.ones(batch_size, seq_len, d_model) * 5.0

        # TODO: IMPLEMENT
        # Layer norm with epsilon for numerical stability
        output = None  # <-- YOUR CODE HERE

        # Verification
        assert output is not None, "Implementation required"
        assert not torch.isnan(output).any(), "Output should not contain NaN"
