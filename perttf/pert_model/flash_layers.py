import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

FLASH_ATTENTION_VERSION = None
flash_attn_qkvpacked_func = None
flash_attn_varlen_func = None

# 2. Try to import the newest version first
try:
    # Assuming 'flash_attn_interface' is the newer package/module
    from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_varlen_func
    FLASH_ATTENTION_VERSION = '3'
    print("✅ Detected Flash Attention v3.")
except ImportError:
    # 3. If the first import fails, try the next one
    try:
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_func
        FLASH_ATTENTION_VERSION = '2'
        print("✅ Detected Flash Attention v2.")
    except ImportError:
        # 4. If all imports fail, provide a notice
        print("⚠️ Flash Attention not installed. Model will use standard attention.")


class Empty:
    pass

class FlashTransformerEncoderLayerVarlen(nn.Module):
    """
    Alternative implementation that uses flash_attn_varlen_func for better handling
    of sequences with different lengths (padding).
    """
    
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
        causal=False,
        bias = True,
        use_flash_attn = True
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.flash_version = FLASH_ATTENTION_VERSION
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.causal = causal
        
        # Multi-head attention components
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.self_attn = Empty() # Dummy code because TransformerEncoder expects self.self_attn.batch_first
        self.self_attn.batch_first = batch_first
        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def _compute_packing_info(self, key_padding_mask):
        """
        Pre-compute all packing information to avoid expensive loops.
        
        Args:
            key_padding_mask: Boolean mask of shape (batch_size, seq_len)
                             True indicates positions to be masked
        
        Returns:
            batch_indices: Tensor of batch indices for valid positions
            seq_indices: Tensor of sequence indices for valid positions  
            seqlens: Tensor of actual sequence lengths per batch
            cu_seqlens: Cumulative sequence lengths for flash attention
            total_valid_tokens: Total number of valid (non-padded) tokens
        """
        valid_mask = ~key_padding_mask  # True for valid positions
        # Find all valid positions at once using vectorized operations
        batch_indices, seq_indices = torch.where(valid_mask)
        
        # Compute actual sequence lengths per batch
        seqlens = valid_mask.sum(dim=1, dtype=torch.int32)
        
        # Create cumulative sequence lengths for flash attention
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=key_padding_mask.device),
            seqlens.cumsum(dim=0, dtype=torch.int32)
        ])
        
        total_valid_tokens = batch_indices.shape[0]
        
        return batch_indices, seq_indices, seqlens, cu_seqlens, total_valid_tokens

    def _pack_sequences_fast(self, tensor, batch_indices, seq_indices):
        """
        Fast packing using advanced indexing instead of loops.
        
        Args:
            tensor: Input tensor of shape (batch_size, seq_len, nhead, head_dim)
            batch_indices: Batch indices for valid positions
            seq_indices: Sequence indices for valid positions
            
        Returns:
            packed_tensor: Tensor of shape (total_valid_tokens, nhead, head_dim)
        """
        # Use advanced indexing - much faster than loops and concatenation
        return tensor[batch_indices, seq_indices]

    def _unpack_sequences_fast(self, packed_tensor, batch_indices, seq_indices, orig_shape):
        """
        Fast unpacking using direct assignment instead of loops.
        
        Args:
            packed_tensor: Packed tensor of shape (total_valid_tokens, nhead, head_dim)
            batch_indices: Batch indices for valid positions
            seq_indices: Sequence indices for valid positions
            orig_shape: Original shape (batch_size, seq_len, nhead, head_dim)
            
        Returns:
            unpacked_tensor: Tensor of original shape with results scattered back
        """
        batch_size, seq_len, nhead, head_dim = orig_shape
        
        # Initialize output tensor with zeros
        output = torch.zeros(
            orig_shape,
            dtype=packed_tensor.dtype,
            device=packed_tensor.device
        )
        
        # Use advanced indexing for fast scattering
        output[batch_indices, seq_indices] = packed_tensor
        
        return output


    def _flash_attention(self, x, key_padding_mask=None):
        """
        Perform flash attention on the input tensor using variable length attention
        when padding mask is present.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            key_padding_mask: Boolean mask of shape (batch_size, seq_len)
                             True indicates positions to be masked
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * d_model)
        
        # Reshape to separate Q, K, V
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        #print(qkv.shape)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # (3, batch_size, seq_len, nhead, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, seq_len, nhead, head_dim)
        #print(qkv.shape)
        # Check if we have any padding
        if key_padding_mask is not None:
            # Convert to boolean if needed
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.bool()
            
            # Check if there's actual padding
            if not key_padding_mask.any():
                key_padding_mask = None
        
        if key_padding_mask is None:
            # No padding mask - use the efficient packed version
            # Repack for flash_attn_qkvpacked_func
            qkv_for_flash = torch.stack([q, k, v], dim=4)  # (batch, seq_len, nhead, head_dim, 3)
            #print(qkv_for_flash.shape)
            qkv_for_flash = qkv_for_flash.permute(0, 1, 4, 2, 3)  # (batch, seq_len, 3, nheads, head_dim)
            #print(qkv_for_flash.shape)
            if self.flash_version == '3':
                attn_output = flash_attn_qkvpacked_func(
                    qkv_for_flash,
                    softmax_scale=None,
                    causal=self.causal,
                )
            elif self.flash_version == '2':
                attn_output = flash_attn_qkvpacked_func(
                    qkv_for_flash,
                    dropout_p=self.dropout1.p if self.training else 0.0,
                    softmax_scale=None,
                    causal=self.causal,
                    return_attn_probs=False,
                )
            #print(attn_output.shape)
            # attn_output is (batch, seq_len, nhead, head_dim)
        else:
            # Use variable length attention for sequences with padding
            # Calculate actual sequence lengths
            #seqlens = (~key_padding_mask).sum(dim=1, dtype=torch.int32)

            batch_indices, seq_indices, seqlens, cu_seqlens, total_valid_tokens = \
                self._compute_packing_info(key_padding_mask)

            # Handle edge case where all sequences might be fully padded
            
            # Create cumulative sequence lengths
            """
            cu_seqlens = torch.cat([
                torch.tensor([0], dtype=torch.int32, device=x.device),
                seqlens.cumsum(dim=0, dtype=torch.int32)
            ])
            
            # Create mask for valid positions
            valid_mask = ~key_padding_mask  # True for valid positions
            
            # Pack sequences by removing padded positions
            q_packed_list = []
            k_packed_list = []
            v_packed_list = []
            
            for b in range(batch_size):
                valid_indices = valid_mask[b]
                if valid_indices.any():  # Only process if there are valid tokens
                    q_packed_list.append(q[b][valid_indices])
                    k_packed_list.append(k[b][valid_indices])
                    v_packed_list.append(v[b][valid_indices])
            """
            # Handle edge case where all sequences might be fully padded
            #if q_packed_list:
            if total_valid_tokens == 0:
                # All sequences are fully padded - return zeros
                attn_output = torch.zeros(
                    batch_size, seq_len, self.nhead, self.head_dim,
                    dtype=x.dtype,
                    device=x.device
                )
            else:
                # Fast packing using vectorized operations
                q_packed = self._pack_sequences_fast(q, batch_indices, seq_indices)
                k_packed = self._pack_sequences_fast(k, batch_indices, seq_indices)
                v_packed = self._pack_sequences_fast(v, batch_indices, seq_indices)
                #q_packed = torch.cat(q_packed_list, dim=0)  # (total_valid_tokens, nhead, head_dim)
                #k_packed = torch.cat(k_packed_list, dim=0)
                #v_packed = torch.cat(v_packed_list, dim=0)
                
                # Apply variable length flash attention
                max_seqlen = int(seqlens.max().item())
                if self.flash_version == '3':
                    attn_output_packed = flash_attn_varlen_func(
                        q_packed,
                        k_packed, 
                        v_packed,
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen,
                        softmax_scale=None,
                        causal=self.causal,
                    )
                elif self.flash_version == '2':
                    attn_output_packed = flash_attn_varlen_func(
                        q_packed,
                        k_packed, 
                        v_packed,
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen,
                        dropout_p=self.dropout1.p if self.training else 0.0,
                        softmax_scale=None,
                        causal=self.causal,
                        return_attn_probs=False,
                    )
                
                # attn_output_packed shape: (total_valid_tokens, nhead, head_dim)
                orig_shape = (batch_size, seq_len, self.nhead, self.head_dim)
                attn_output = self._unpack_sequences_fast(
                    attn_output_packed, 
                    batch_indices, 
                    seq_indices, 
                    orig_shape
                )
                
                # OLD: Unpack the output back to original shape with padding
                """
                attn_output = torch.zeros(
                    batch_size, seq_len, self.nhead, self.head_dim,
                    dtype=attn_output_packed.dtype,
                    device=attn_output_packed.device
                )
                
                # Scatter the results back to their original positions
                start_idx = 0
                for b in range(batch_size):
                    valid_indices = valid_mask[b]
                    if valid_indices.any():
                        num_valid = valid_indices.sum().item()
                        attn_output[b][valid_indices] = attn_output_packed[start_idx:start_idx + num_valid]
                        start_idx += num_valid
                
            else:
                # All sequences are fully padded
                attn_output = torch.zeros(
                    batch_size, seq_len, self.nhead, self.head_dim,
                    dtype=x.dtype,
                    device=x.device
                )
                """
                
        # Reshape output: (batch, seq_len, nhead, head_dim) -> (batch, seq_len, d_model)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        return self.out_proj(attn_output)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
                Shape: (batch_size, seq_len, d_model) if batch_first=True
            src_mask: the mask for the src sequence (optional).
                Note: FlashAttention v2 has limited support for arbitrary attention masks
            src_key_padding_mask: the mask for the src keys per batch (optional).
                Shape: (batch_size, seq_len), True means ignore/mask that position
                Can be bool or float tensor (will be converted to bool)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        
        if src_mask is not None:
            # FlashAttention v2 supports causal masks natively but arbitrary masks need special handling
            if not self.causal:
                raise ValueError(
                    "FlashAttention v2 only supports causal masks natively. "
                    "For arbitrary attention masks, consider using standard attention."
                )
        
        # Ensure batch_first format
        if not self.batch_first:
            src = src.transpose(0, 1)
        
        if self.norm_scheme == "pre":
            # Pre-normalization
            src = self.norm1(src)
            src2 = self._flash_attention(src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            # Post-normalization
            src2 = self._flash_attention(src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
        # Convert back if needed
        if not self.batch_first:
            src = src.transpose(0, 1)
            
        return src
    

# Try to use other backends for attention
class SDPATransformerEncoderLayer(nn.Module):
    """
    A Transformer Encoder Layer that uses torch.nn.functional.scaled_dot_product_attention
    to automatically leverage the best available attention backend (e.g., FlashAttention, cuDNN).
    """
    
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",
        causal=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.causal = causal
        self.self_attn = Empty()
        self.self_attn.batch_first = batch_first
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        # REMOVED: The FlashSelfAttention module and use_flash_attn flag are no longer needed.
        
        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Layer normalization and dropouts
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    # SIMPLIFIED: Replaced the complex _flash_attention method with a cleaner one.
    def _attention(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Perform attention using PyTorch's scaled_dot_product_attention.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * d_model)
        
        # Reshape and permute for SDPA
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, nhead, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, nhead, seq_len, head_dim)
        
        attn_mask = key_padding_mask
        if attn_mask is not None:
            # attn_mask must be broadcastable to (batch, nhead, seq_len, seq_len)
            # We add the nhead and query_seq_len dimensions.
            attn_mask = attn_mask.view(batch_size, 1, 1, seq_len)


        # The entire logic for varlen and packed attention is replaced by this single call.
        # SDPA handles the padding mask and causality internally.
        with torch.nn.attention.sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,  # Pass the reshaped mask here
                dropout_p=self.dropout1.p if self.training else 0.0,
                is_causal=self.causal
            )
            
        # Reshape output back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_output)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if src_mask is not None and not self.causal:
            raise ValueError("SDPATransformerEncoderLayer only supports a causal mask via the 'causal' flag.")
        
        if not self.batch_first:
            src = src.transpose(0, 1)
        
        if self.norm_scheme == "pre":
            src_norm = self.norm1(src)
            attn_out = self._attention(src_norm, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(attn_out)
            
            src_norm = self.norm2(src)
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
            src = src + self.dropout2(ff_out)
        else: # post-norm
            attn_out = self._attention(src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(attn_out)
            src = self.norm1(src)
            
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(ff_out)
            src = self.norm2(src)
        
        if not self.batch_first:
            src = src.transpose(0, 1)
            
        return src

class FlashCrossTransformerLayer(nn.Module):
    """
    A Transformer Layer specifically designed for Cross Attention between:
    1. Dense Queries (Vocab Embeddings) - Uniform length, no padding.
    2. Ragged Keys/Values (Context) - Variable length, has padding.
    
    It uses FlashAttention Varlen to skip computation on padding tokens in the Context.
    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        device=None,
        batch_first=True,
        dtype=None,
        norm_scheme="post", # "pre" or "post"
        bias=False
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.flash_version = FLASH_ATTENTION_VERSION
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        self.dropout_p = dropout
        self.norm_scheme = norm_scheme
        self.self_attn = Empty() # Dummy code because TransformerEncoder expects self.self_attn.batch_first
        self.self_attn.batch_first = batch_first
        # 1. Cross Attention Components
        # We separate Q from KV because they come from different tensors
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        # 2. Feed Forward Components
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu": return F.relu
        elif activation == "gelu": return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def _pack_memory_kv(self, key_input, padding_mask):
        """
        Packs the context (Key/Value) input by removing padded tokens.
        Returns packed KV tensor and cu_seqlens_k.
        """
        batch_size, seq_len, _ = key_input.shape
        
        # Project to K, V: (B, L, 2*D) -> (B, L, 2, H, D_head)
        kv = self.kv_proj(key_input).view(batch_size, seq_len, 2, self.nhead, self.head_dim)
        k, v = kv.unbind(2) # Each is (B, L, H, D_head)

        if padding_mask is None:
            # No padding: Simple flatten
            k_packed = k.reshape(-1, self.nhead, self.head_dim)
            v_packed = v.reshape(-1, self.nhead, self.head_dim)
            
            # cu_seqlens is simple arithmetic progression: [0, L, 2L, ...]
            seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=key_input.device)
            cu_seqlens_k = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=key_input.device),
                seqlens.cumsum(0, dtype=torch.int32)
            ])
            max_seqlen_k = seq_len
        else:
            # Handle Padding: Remove masked tokens
            valid_mask = ~padding_mask.bool() # True = Valid Token
            seqlens = valid_mask.sum(dim=1, dtype=torch.int32)
            
            cu_seqlens_k = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=key_input.device),
                seqlens.cumsum(0, dtype=torch.int32)
            ])
            
            # Pack using boolean indexing
            valid_indices = torch.nonzero(valid_mask.flatten(), as_tuple=True)[0]
            
            k_flat = k.view(-1, self.nhead, self.head_dim)
            v_flat = v.view(-1, self.nhead, self.head_dim)
            
            k_packed = k_flat[valid_indices]
            v_packed = v_flat[valid_indices]
            max_seqlen_k = seqlens.max().item()

        return k_packed, v_packed, cu_seqlens_k, max_seqlen_k

    def _cross_attention(self, tgt, memory, memory_padding_mask=None):
        """
        tgt: (Batch, Vocab, Dim) - The Query
        memory: (Batch, Context, Dim) - The Key/Value
        """
        batch_size, vocab_size, _ = tgt.shape
        
        # 1. Prepare Queries (Dense)
        # (B, V, D) -> (B, V, H, D_head) -> (B*V, H, D_head)
        q = self.q_proj(tgt).view(batch_size, vocab_size, self.nhead, self.head_dim)
        q_packed = q.view(-1, self.nhead, self.head_dim)
        
        # Create cu_seqlens for Q (Uniform length)
        # [0, V, 2V, ..., B*V]
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * vocab_size, step=vocab_size, 
            dtype=torch.int32, device=tgt.device
        )
        max_seqlen_q = vocab_size

        # 2. Prepare Keys/Values (Ragged)
        k_packed, v_packed, cu_seqlens_k, max_seqlen_k = self._pack_memory_kv(memory, memory_padding_mask)

        # 3. Flash Attention Varlen
        # Note: dropout_p=0.0 during inference
        if self.flash_version == '3':
                attn_out_packed = flash_attn_varlen_func(
                q_packed, k_packed, v_packed,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=None,
                causal=False
            )
        
        elif self.flash_version == '2':
            attn_out_packed = flash_attn_varlen_func(
                q_packed, k_packed, v_packed,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=None,
                causal=False
            )
        

        # 4. Unpack and Project
        # (B*V, H, D_head) -> (B, V, D)
        attn_out = attn_out_packed.view(batch_size, vocab_size, self.d_model)
        return self.out_proj(attn_out)

    def forward(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor, 
        memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: (Batch, Vocab_Size, Dim) - The static embeddings (Query Source)
            memory: (Batch, Seq_Len, Dim) - The transformer output (Key/Value Source)
            memory_key_padding_mask: (Batch, Seq_Len) - True where padding exists
        
        Returns:
            (Batch, Vocab_Size, Dim) - The updated embeddings
        """
        
        # Pre-Norm Architecture
        if self.norm_scheme == "pre":
            # 1. Cross Attention Block
            tgt_norm = self.norm1(tgt)
            attn_out = self._cross_attention(tgt_norm, memory, memory_key_padding_mask)
            tgt = tgt + self.dropout1(attn_out)
            
            # 2. Feed Forward Block
            tgt_norm = self.norm2(tgt)
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
            tgt = tgt + self.dropout2(ff_out)
            
        # Post-Norm Architecture
        else:
            # 1. Cross Attention Block
            attn_out = self._cross_attention(tgt, memory, memory_key_padding_mask)
            tgt = self.norm1(tgt + self.dropout1(attn_out))
            
            # 2. Feed Forward Block
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = self.norm2(tgt + self.dropout2(ff_out))
            
        return tgt