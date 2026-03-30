import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import triton.language as tl

from typing import Optional
from kernel import fused_qk_attention, fused_apply_rotary_emb, fused_rms_norm, fused_mask_softmax
from cache_manager import PageAttentionCacheManager


def precompute_freqs_cis(
    qk_rope_head_dim: int, 
    seq_len: int, 
    seq_len_train: int, 
    beta_fast: int, 
    beta_slow: int, 
    rope_theta: float, 
    rope_factor: float,
    dtype: Optional[torch.dtype] = torch.float16
) -> torch.Tensor:
    """
    This function is adapted from 
    https://github.com/deepseek-ai/DeepSeek-V3/blob/a878eada08ea6913f5a2ae80a43afeffdef082ef/inference/model.py#L294

    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        seq_len: the seq len used during inference
        seq_len_train: the seq len used during training

        rope_theta: the base theta


    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # theta_i = 1 / (rope_theta ^ (2i / d)) i in {0, 2, ... d - 2}
    freqs = 1.0 / (rope_theta ** (torch.arange(0, qk_rope_head_dim, 2) / qk_rope_head_dim)) # [D // 2]
    if seq_len > seq_len_train: # if inference seq_len > seq_len_train
        low, high = find_correction_range(beta_fast, beta_slow, qk_rope_head_dim, rope_theta, seq_len_train)
        smooth = 1 - linear_ramp_factor(low, high, qk_rope_head_dim // 2)
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seq_len) # [L]
    freqs = torch.outer(t, freqs) # outer product, freqs[t, i] = theta_i * t [L, D // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # exp(j * theta_i * t) [L, D // 2], but in complex form, torch.complex64 if input float32
    return torch.view_as_real(freqs_cis).to(dtype) # [L, D // 2, 2], 0 for real, 1 for imag


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed values (with real and image part of the original complex value) for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    # x: [batch_size, seq_len, num_heads, qk_rope_head_dim (D)]
    # freqs_cis: [L, D // 2, 2]
    dtype = x.dtype
    # torch.view_as_real(torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))) == x.view(*x.shape[:-1], -1, 2)
    x = torch.view_as_real(torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))) # [..., D // 2, 2]
    x_real = x[..., 0] # [... , D // 2]
    x_imag = x[..., 1] # [... , D // 2]
    freqs_cos = freqs_cis[:, :, 0].unsqueeze(0).unsqueeze(2) # [1, L, 1, D // 2]
    freqs_sin = freqs_cis[:, :, 1].unsqueeze(0).unsqueeze(2) # [1, L, 1, D // 2]

    out_real = x_real * freqs_cos - x_imag * freqs_sin # [batch_size, seq_len, num_heads, D // 2]
    out_imag = x_real * freqs_sin + x_imag * freqs_cos
    # [out_real, out_imag]: [batch_size, seq_len, num_heads, D // 2, 2]
    # torch stack(): [batch_size, seq_len, num_heads, D // 2, 2]
    # flatten(3): [batch_size, seq_len, num_heads, D], [real0, imag0, real1, imag1, ...], interleave, rather than concat
    out = torch.stack([out_real, out_imag], dim=-1).flatten(3)

    assert out.dtype == dtype, f"Output dtype {out.dtype} does not match input dtype {dtype}"
    return out

def apply_rotary_emb_origin(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    THIS IS THE ORIGINAL VERSION OF APPLY_ROTARY_EMB BASED ON COMPLEX FREQS_CIS
    BUT I MODIFIED IT TO USE THE 2 SEPARATE DIMENSIONS FOR REAL AND IMAGINARY PARTS
    DO NOT USE THIS FUNCTION !!!!!
    """
    # x: [batch_size, seq_len, num_heads, qk_rope_head_dim (D)]
    # freqs_cis: [L, D // 2]
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2)) # [B, L, H, D // 2]
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1)) # [1, L, 1, D // 2]
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

class MLA(nn.Module):
    def __init__(
        self,
        dim: int, # model dim
        kv_latent_rank: int, # rank of the cached compressed kv (c_kv)
        q_latent_rank: int, # rank of the cached compressed q (c_q)
        num_heads: int, # number of heads
        qk_nrope_head_dim: int, # dim of the q and k heads
        v_head_dim: int, # dim of the v head
        qk_rope_head_dim: int, # dim of the q and k rotary embeddings
        max_batch_size: int, # max batch size
        max_seq_len: int, # max sequence length
        dtype: Optional[torch.dtype] = torch.float16, # the dtype of the model
        optim_type: str = "torch", # the type of the optimization, "torch" or "triton",
        eps: float = 1e-6, # epsilon for RMSNorm
        use_page_cache: bool = False,
        use_page_cache_triton: bool = False,
        page_size: int = 128,
    ):
        super().__init__()
        
        self.optim_type = optim_type
        
        self.qk_head_dim = qk_nrope_head_dim + qk_rope_head_dim
        self.proj_kv_down = nn.Linear(dim, kv_latent_rank + qk_rope_head_dim, bias=False, dtype=dtype)
        self.proj_kv_up = nn.Linear(kv_latent_rank, num_heads * (qk_nrope_head_dim + v_head_dim), bias=False, dtype=dtype)
        self.proj_q_down = nn.Linear(dim, q_latent_rank, bias=False, dtype=dtype)
        self.proj_q_up = nn.Linear(q_latent_rank, num_heads * self.qk_head_dim, bias=False, dtype=dtype)
        self.proj_out = nn.Linear(num_heads * v_head_dim, dim, bias=False, dtype=dtype)
        
        self.rms_norm_kv_weight = torch.nn.Parameter(torch.ones(kv_latent_rank, dtype=dtype))
        self.rms_norm_q_weight = torch.nn.Parameter(torch.ones(q_latent_rank, dtype=dtype))

        self.softmax_scale = 1.0 / self.qk_head_dim ** 0.5
        self.num_heads = num_heads
        self.kv_latent_rank = kv_latent_rank
        self.q_latent_rank = q_latent_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nrope_head_dim = qk_nrope_head_dim
        self.v_head_dim = v_head_dim
        
        self.eps = eps
        self.use_page_cache = use_page_cache
        self.use_page_cache_triton = use_page_cache_triton
        self.page_size = page_size
        
        if not use_page_cache:
            self.register_buffer(
                "kv_latent_cache", 
                torch.zeros(max_batch_size, max_seq_len, kv_latent_rank, dtype=dtype), 
                persistent=False
            )
            self.register_buffer(
                "k_rope_cache", 
                torch.zeros(max_batch_size, max_seq_len, qk_rope_head_dim, dtype=dtype),
                persistent=False
            )
        else:
            num_pages = math.ceil(max_batch_size * max_seq_len / page_size * 1.1)
            
            self.cache_manager = PageAttentionCacheManager(
                batch_size=max_batch_size,
                page_size=page_size,
                num_pages=num_pages,
                kv_latent_rank=kv_latent_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                max_seq_len=max_seq_len,
                use_triton=use_page_cache_triton,
                dtype=dtype,
                device='cuda',
            )

    
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, # the position of the x to be placed on the cache
        freq_cis: torch.Tensor, # the precomputed freq cis for rotary embeddings
        mask: Optional[torch.Tensor] = None, # the mask for the attention, [seq_len_q, seq_len_k]
        return_debug: bool = False,
    ):
        batch_size, seq_len, dim = x.shape
        
        q_latent = self.proj_q_down(x)
        if self.optim_type == "triton" or ("ablation" in self.optim_type and "rmsnorm" in self.optim_type): 
            q_latent = fused_rms_norm(q_latent, (q_latent.shape[-1],), self.rms_norm_q_weight, self.eps)
        else:
            q_latent = torch.nn.functional.rms_norm(q_latent, (q_latent.shape[-1],), self.rms_norm_q_weight, self.eps)
        q = self.proj_q_up(q_latent)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nrope, q_rope = q.split(
            [self.qk_nrope_head_dim, self.qk_rope_head_dim], dim=-1
        ) # q_nrope: [batch_size, seq_len, num_heads, qk_nrope_head_dim], 
        # q_rope: [batch_size, seq_len, num_heads, qk_rope_head_dim]
        if self.optim_type == "torch":
            q_rope = apply_rotary_emb(q_rope, freq_cis) # [batch_size, seq_len, num_heads, qk_rope_head_dim]
        elif (self.optim_type == "triton") or ("ablation" in self.optim_type and "rope" in self.optim_type):
            q_rope = fused_apply_rotary_emb(q_rope, freq_cis)
        else: # default
            q_rope = apply_rotary_emb(q_rope, freq_cis)
        q = torch.cat([q_nrope, q_rope], dim=-1)
        
        kv_latent, k_rope = self.proj_kv_down(x).split(
            [self.kv_latent_rank, self.qk_rope_head_dim], dim=-1
        ) # kv_latent: [batch_size, seq_len, kv_latent_rank], k_rope: [batch_size, seq_len, qk_rope_head_dim]
        if self.optim_type == "torch":
            k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freq_cis).squeeze(2) # [batch_size, seq_len, qk_rope_head_dim]
        elif (self.optim_type == "triton") or ("ablation" in self.optim_type and "rope" in self.optim_type):
            k_rope = fused_apply_rotary_emb(k_rope.unsqueeze(2), freq_cis).squeeze(2)
        else: # default
            k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freq_cis).squeeze(2)

        end_pos = start_pos + seq_len
        if self.optim_type == "triton" or ("ablation" in self.optim_type and "rmsnorm" in self.optim_type): 
            normalized_kv_latent = fused_rms_norm(kv_latent, (kv_latent.shape[-1], ), self.rms_norm_kv_weight, self.eps)
        else:
            normalized_kv_latent = torch.nn.functional.rms_norm(kv_latent, (kv_latent.shape[-1], ), self.rms_norm_kv_weight, self.eps)
            
        if self.use_page_cache:
            if self.use_page_cache_triton:
                self.cache_manager.update_batch(
                    start_pos,
                    normalized_kv_latent,
                    k_rope
                )
            else:
                for b_idx in range(batch_size):
                    self.cache_manager.update(
                        batch_idx=b_idx,
                        start_pos=start_pos,
                        kv_latent=normalized_kv_latent[b_idx],
                        k_rope=k_rope[b_idx]
                    )
        else:
            self.kv_latent_cache[:batch_size, start_pos:end_pos] = normalized_kv_latent
            self.k_rope_cache[:batch_size, start_pos:end_pos] = k_rope

        # reshape the kv up weight, to make it be absorbed into the q
        proj_kv_up_weight = self.proj_kv_up.weight # [num_heads * (qk_nrope_head_dim + v_head_dim, kv_latent_rank]
        proj_kv_up_weight = proj_kv_up_weight.view(
            self.num_heads, self.qk_nrope_head_dim + self.v_head_dim, self.kv_latent_rank
        )
        # q_nrope absorb the kv_up weight, make q_nrope could directly matmul with kv_latent, 
        # and kv_latent can be directly get from the cache
        proj_kv_up_weight_q_nrope_absorbed = proj_kv_up_weight[:, :self.qk_nrope_head_dim, :]
        """
        The motivation of first aborbing the kv_up weight into q_nrope: 
        1. If first upsample the kv_latent, then will cause large k, k is with 
        higher dimension than kv_latent, which causes higher memory usage.
        """
        q_nrope_absorb = torch.einsum(
            "blhd,hdk->blhk", q_nrope, proj_kv_up_weight_q_nrope_absorbed
        ) # [batch_size, seq_len, num_heads, kv_latent_rank]

        if self.use_page_cache:
            # first retrieve all cache data and stack them into a single tensor
            
            if self.use_page_cache_triton:
                stacked_kv_latent, stacked_k_rope = self.cache_manager.retrieve_batch(
                    batch_size,
                    start_pos,
                    end_pos
                )
            else:
                all_kv_latent = []
                all_k_rope = []
                for b_idx in range(batch_size):
                    batch_kv_latent, batch_k_rope = self.cache_manager.retrieve(
                        batch_idx=b_idx,
                        start_pos=0,
                        end_pos=end_pos
                    )
                    all_kv_latent.append(batch_kv_latent)
                    all_k_rope.append(batch_k_rope)
            
                stacked_kv_latent = torch.stack(all_kv_latent, dim=0) # [batch_size, seq_len, kv_latent_rank]
                stacked_k_rope = torch.stack(all_k_rope, dim=0)  # [batch_size, seq_len, qk_rope_head_dim]
            
            if (self.optim_type == "triton") or ("ablation" in self.optim_type and "qk_attention" in self.optim_type):
                kernel_dtype = tl.float16 if x.dtype == torch.float16 else tl.float32
                scores = fused_qk_attention(
                    q_nrope_absorb, q_rope, 
                    stacked_kv_latent, stacked_k_rope,
                    self.softmax_scale, kernel_version=2, dtype=kernel_dtype
                )
            else:
                scores = (
                    torch.einsum("blhk,btk->blht", q_nrope_absorb, stacked_kv_latent) + 
                    torch.einsum("blhr,btr->blht", q_rope, stacked_k_rope)
                ) * self.softmax_scale
        else:
            if (self.optim_type == "triton") or ("ablation" in self.optim_type and "qk_attention" in self.optim_type):
                kernel_dtype = tl.float16 if x.dtype == torch.float16 else tl.float32
                scores = fused_qk_attention(
                    q_nrope_absorb, q_rope, 
                    self.kv_latent_cache[:batch_size, :end_pos], self.k_rope_cache[:batch_size, :end_pos],
                    self.softmax_scale, kernel_version=2, dtype=kernel_dtype
                )
            else:
                scores = (
                    torch.einsum(
                        "blhk,btk->blht", q_nrope_absorb, self.kv_latent_cache[:batch_size, :end_pos]
                    ) + 
                    torch.einsum(
                        "blhr,btr->blht", q_rope, self.k_rope_cache[:batch_size, :end_pos]
                    )
                ) * self.softmax_scale # [batch_size, seq_len_q, num_heads, seq_len_k]

        # mask the scores
        if mask is not None:
            if self.optim_type == "triton" or ("ablation" in self.optim_type and "softmax" in self.optim_type): 
                mask = mask.unsqueeze(1).unsqueeze(0)
                attn_logits = scores
                fused_mask_softmax(scores, mask)

            else:
                mask = mask.unsqueeze(1).unsqueeze(0) # [1, seq_len_q, 1, seq_len_k]
                scores += mask # [batch_size, seq_len_q, num_heads, seq_len_k]
                attn_logits = scores
                scores = scores.softmax(dim=-1)
        else:
            attn_logits = scores
            scores = scores.softmax(dim=-1)

        # matmul cache first, then upsample the v, this reduces the computation, otherwise
        # matmul on the upsampled v, which is much higher dimensional
        if self.use_page_cache:
            # reuse the already stacked kv_latent from the previous step
            latent_out = torch.einsum(
                "blht,btk->blhk", scores, stacked_kv_latent
            )
        else:
            latent_out = torch.einsum(
                "blht,btk->blhk", scores, self.kv_latent_cache[:batch_size, :end_pos]
            )
        proj_kv_up_weight_v = proj_kv_up_weight[:, -self.v_head_dim:, :]
        x = torch.einsum(
            "blhk,hdk->blhd", latent_out, proj_kv_up_weight_v
        ) # [batch_size, seq_len, num_heads, v_head_dim]

        x = x.flatten(start_dim=2) # [batch_size, seq_len, num_heads * v_head_dim]
        x = self.proj_out(x)

        if return_debug:
            debug_dict = {
                "hidden": x,
                "latent": latent_out,
                "scores": scores,  # 注意：这里是 softmax 后的 probs
                "q_nrope_absorb": q_nrope_absorb,
                "q_rope": q_rope,
                "normalized_kv_latent": normalized_kv_latent,
                "k_rope": k_rope,
                "attn_logits": attn_logits,
            }
            if self.use_page_cache:
                debug_dict["stacked_kv_latent"] = stacked_kv_latent
                debug_dict["stacked_k_rope"] = stacked_k_rope
            else:
                debug_dict["stacked_kv_latent"] = self.kv_latent_cache[:batch_size, :end_pos]
                debug_dict["stacked_k_rope"] = self.k_rope_cache[:batch_size, :end_pos]
            return debug_dict
        return x

class FFN(nn.Module):
    def __init__(
        self, 
        dim: int, # model dim
        ffn_hidden_dim: int, # hidden dim of the ffn
    ):
        super().__init__()
        self.proj_up_1 = nn.Linear(dim, ffn_hidden_dim, bias=False)
        self.proj_up_2 = nn.Linear(dim, ffn_hidden_dim, bias=False)
        self.proj_down = nn.Linear(ffn_hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.proj_down(F.silu(self.proj_up_1(x) + self.proj_up_2(x)))
    
class Layer(nn.Module):
    def __init__(
        self, 
        dim: int, # model dim
        kv_latent_rank: int, # rank of the cached compressed kv (c_kv)
        q_latent_rank: int, # rank of the cached compressed q (c_q)
        num_heads: int, # number of heads
        qk_nrope_head_dim: int, # dim of the q and k heads
        v_head_dim: int, # dim of the v head
        qk_rope_head_dim: int, # dim of the q and k rotary embeddings
        max_batch_size: int, # max batch size
        max_seq_len: int, # max sequence length
        ffn_hidden_dim: int, # hidden dim of the ffn
    ):
        super().__init__()
        self.mla = MLA(
            dim, kv_latent_rank, q_latent_rank, num_heads, qk_nrope_head_dim, 
            v_head_dim, qk_rope_head_dim, max_batch_size, max_seq_len
        )
        self.ffn = FFN(dim, ffn_hidden_dim)
        self.mla_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)

    def forward(
            self, 
            x: torch.Tensor, 
            start_pos: int, 
            freq_cis: torch.Tensor, 
            mask: Optional[torch.Tensor] = None
        ):
        # use prenorm
        x = x + self.mla(self.mla_norm(x), start_pos, freq_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self, 
        dim: int, # model dim
        kv_latent_rank: int, # rank of the cached compressed kv (c_kv)
        q_latent_rank: int, # rank of the cached compressed q (c_q)
        num_heads: int, # number of heads
        qk_nrope_head_dim: int, # dim of the q and k heads
        v_head_dim: int, # dim of the v head
        qk_rope_head_dim: int, # dim of the q and k rotary embeddings
        max_batch_size: int, # max batch size
        max_seq_len: int, # max sequence length
        max_seq_len_train: int, # max seq len used during training
        ffn_hidden_dim: int, # hidden dim of the ffn
        num_layers: int, # number of layers
        vocab_size: int, # vocabulary size
    ): 
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            Layer(
                dim, kv_latent_rank, q_latent_rank, num_heads, 
                qk_nrope_head_dim, v_head_dim, qk_rope_head_dim, 
                max_batch_size, max_seq_len, ffn_hidden_dim, 
            ) for _ in range(num_layers)
        ])
        self.register_buffer(
            "freq_cis", 
            precompute_freqs_cis(
                qk_rope_head_dim, max_seq_len, max_seq_len_train, 
                beta_fast=32, beta_slow=1, rope_theta=10000.0, 
                rope_factor=40.0
            )
        )
        self.final_norm = nn.RMSNorm(dim)
        self.final_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self, 
        tokens: torch.Tensor, # [batch_size, seq_len]
        start_pos: int = 0,
    ):
        x = self.embedding(tokens)
        seq_len = tokens.shape[1]
        end_pos = start_pos + seq_len
        freq_cis = self.freq_cis[start_pos:end_pos]
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, start_pos + seq_len), float("-inf"), device=tokens.device).triu_(1)
        
        for layer in self.layers:
            x = layer(x, start_pos, freq_cis, mask)
        
        x = self.final_norm(x)[:, -1] # use the last as the prediction
        logits = self.final_proj(x)
        return logits

if __name__ == "__main__":
    torch.manual_seed(42)

    dim = 64
    kv_latent_rank = 32
    q_latent_rank = 32
    num_heads = 4
    qk_nrope_head_dim = 24
    v_head_dim = 16
    qk_rope_head_dim = 40
    max_batch_size = 2
    max_seq_len = 16
    max_seq_len_train = 8
    ffn_hidden_dim = 128
    num_layers = 2
    vocab_size = 100

    model = Transformer(
        dim=dim,
        kv_latent_rank=kv_latent_rank,
        q_latent_rank=q_latent_rank,
        num_heads=num_heads,
        qk_nrope_head_dim=qk_nrope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        max_seq_len_train=max_seq_len_train,
        ffn_hidden_dim=ffn_hidden_dim,
        num_layers=num_layers,
        vocab_size=vocab_size
    )

    model.eval()
    batch_size = 2
    seq_len = 5
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Tokens shape: {tokens.shape}")

    with torch.no_grad():
        logits = model(tokens, start_pos=0)

    print("Input shape:", tokens.shape)               # [2, 5]
    print("Logits shape:", logits.shape)              # [2, vocab_size]
    assert logits.shape == (batch_size, vocab_size), "Output shape mismatch"
    print("Test passed.")
