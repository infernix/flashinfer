import math
import pytest
import torch

import flashinfer
from flashinfer import BatchPrefillWithPagedKVCacheWrapper


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_scale_forwarding_effect(dtype):
    torch.manual_seed(42)

    H_QO, H_KV, N_CTX, HEAD_DIM, PAGE_SIZE = 1, 1, 8, 64, 16
    max_num_pages = (N_CTX + PAGE_SIZE - 1) // PAGE_SIZE

    # Create paged KV cache
    k_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    v_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    paged_kv_cache = (k_cache, v_cache)

    # Create query tensor and indptrs
    q = torch.randn(N_CTX, H_QO, HEAD_DIM, dtype=dtype, device="cuda")
    qo_indptr = torch.tensor([0, N_CTX], dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor(
        [N_CTX % PAGE_SIZE or PAGE_SIZE], dtype=torch.int32, device="cuda"
    )

    workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)

    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    out1, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=0.1, v_scale=0.1)
    out2, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=2.0, v_scale=2.0)

    assert not torch.allclose(out1, out2, atol=1e-3), (
        "Output should change when k_scale/v_scale values are different. "
        "This may indicate that the arguments are not passed correctly."
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_scale_forwarding_math_property(dtype: torch.dtype):
    torch.manual_seed(0)

    # ---------------- parameters ----------------
    N_CTX, PAGE_SIZE = 128, 16
    H_QO, H_KV, HEAD_DIM = 1, 1, 64  # Explicitly specify H_QO
    max_num_pages = (N_CTX + PAGE_SIZE - 1) // PAGE_SIZE

    # ---------------- paged KV cache ----------------
    k_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    v_cache = torch.randn_like(k_cache)
    paged_kv_cache = (k_cache, v_cache)

    # ---------------- query and indptr ----------------
    q = torch.randn(N_CTX, H_QO, HEAD_DIM, dtype=dtype, device="cuda")
    qo_indptr = torch.tensor([0, N_CTX], dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor(
        [N_CTX % PAGE_SIZE or PAGE_SIZE], dtype=torch.int32, device="cuda"
    )

    # ---------------- wrapper ----------------
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)

    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # ---------------- scale factors ----------------
    k_scale = 0.5
    v_scale = 2.0

    # -------- case 1: k_scale only ----------
    out1, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=k_scale)
    out1_ref, _ = wrapper.forward_return_lse(q * k_scale, paged_kv_cache)
    torch.testing.assert_close(out1, out1_ref, rtol=1e-2, atol=1e-3)

    # -------- case 2: v_scale only ----------
    out2, _ = wrapper.forward_return_lse(q, paged_kv_cache, v_scale=v_scale)
    out2_ref, _ = wrapper.forward_return_lse(q, paged_kv_cache)
    torch.testing.assert_close(out2, out2_ref * v_scale, rtol=1e-2, atol=1e-3)

    # -------- case 3: both k_scale and v_scale ----------
    out3, _ = wrapper.forward_return_lse(
        q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale
    )
    out3_ref, _ = wrapper.forward_return_lse(q * k_scale, paged_kv_cache)
    torch.testing.assert_close(out3, out3_ref * v_scale, rtol=1e-2, atol=1e-3)


def _paged_attention_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    page_size = k_cache.shape[1]
    num_qo_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    group_size = num_qo_heads // num_kv_heads
    outputs = []

    qo_indptr_cpu = qo_indptr.cpu()
    paged_kv_indptr_cpu = paged_kv_indptr.cpu()
    paged_kv_indices_cpu = paged_kv_indices.cpu()
    paged_kv_last_page_len_cpu = paged_kv_last_page_len.cpu()

    for request_idx in range(len(qo_indptr_cpu) - 1):
        q_begin = qo_indptr_cpu[request_idx].item()
        q_end = qo_indptr_cpu[request_idx + 1].item()
        page_begin = paged_kv_indptr_cpu[request_idx].item()
        page_end = paged_kv_indptr_cpu[request_idx + 1].item()
        page_indices = paged_kv_indices_cpu[page_begin:page_end].to(q.device)
        kv_len = (page_end - page_begin - 1) * page_size
        kv_len += paged_kv_last_page_len_cpu[request_idx].item()

        k = k_cache.index_select(0, page_indices).reshape(-1, num_kv_heads, k_cache.shape[-1])[
            :kv_len
        ]
        v = v_cache.index_select(0, page_indices).reshape(-1, num_kv_heads, v_cache.shape[-1])[
            :kv_len
        ]
        k = k.repeat_interleave(group_size, dim=1).float()
        v = v.repeat_interleave(group_size, dim=1).float()
        qi = q[q_begin:q_end].float()

        scores = torch.einsum("qhd,khd->hqk", qi, k) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khv->qhv", probs, v).to(v_cache.dtype))

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_tensor_core_decode_falls_back_for_large_head_dim_bfloat16(use_cuda_graph):
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device)[0] != 8:
        pytest.skip("regression targets FA2 on Ampere/Ada GPUs")

    torch.manual_seed(0)
    batch_size = 2
    kv_len = 8
    page_size = 1
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 512
    dtype = torch.bfloat16
    total_num_pages = batch_size * kv_len

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        total_num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32)
    paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * kv_len
    paged_kv_indices = torch.arange(total_num_pages, dtype=torch.int32)
    paged_kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    if use_cuda_graph:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer,
            "NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=torch.empty(
                batch_size + 1, dtype=torch.int32, device=device
            ),
            paged_kv_indices_buffer=torch.empty(
                total_num_pages, dtype=torch.int32, device=device
            ),
            paged_kv_last_page_len_buffer=torch.empty(
                batch_size, dtype=torch.int32, device=device
            ),
            use_tensor_cores=True,
        )
    else:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer,
            "NHD",
            use_tensor_cores=True,
        )

    wrapper.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    out = wrapper.run(q, (k_cache, v_cache))
    ref = _paged_attention_ref(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        1.0 / math.sqrt(head_dim),
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_paged_prefill_selects_cudnn_for_large_head_dim(use_cuda_graph):
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device)[0] != 8:
        pytest.skip("regression targets non-hopper FA2 devices")

    batch_size = 2
    qo_len = 4
    kv_len = 8
    page_size = 1
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 512

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * kv_len
    paged_kv_indices = torch.arange(batch_size * kv_len, dtype=torch.int32)
    paged_kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    if use_cuda_graph:
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            use_cuda_graph=True,
            qo_indptr_buf=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
            paged_kv_indptr_buf=torch.empty(
                batch_size + 1, dtype=torch.int32, device=device
            ),
            paged_kv_indices_buf=torch.empty(
                batch_size * kv_len, dtype=torch.int32, device=device
            ),
            paged_kv_last_page_len_buf=torch.empty(
                batch_size, dtype=torch.int32, device=device
            ),
            backend="auto",
        )
    else:
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            backend="auto",
        )

    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        head_dim_vo=head_dim,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        causal=True,
    )

    expected_qo_indptr_last = int(qo_indptr[-1]) * num_qo_heads * head_dim
    assert wrapper._backend == "cudnn"
    assert wrapper._qo_indptr_last == expected_qo_indptr_last
    assert wrapper._qo_indptr_buf[-1].item() == expected_qo_indptr_last
    assert wrapper._block_tables is not None
    assert wrapper._block_tables.shape == (batch_size, kv_len)


def test_paged_prefill_cudnn_fallback_rejects_sinks():
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device)[0] != 8:
        pytest.skip("regression targets non-hopper FA2 devices")

    batch_size = 2
    qo_len = 4
    kv_len = 8
    page_size = 1
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 512
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, backend="auto")
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * kv_len
    paged_kv_indices = torch.arange(batch_size * kv_len, dtype=torch.int32)
    paged_kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32)
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(
        batch_size * kv_len, page_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)

    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        head_dim_vo=head_dim,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        causal=True,
    )

    with pytest.raises(NotImplementedError, match="does not support sinks"):
        wrapper.run(q, (k_cache, v_cache), sinks=torch.zeros(1, device=device))
