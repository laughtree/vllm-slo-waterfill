<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## SLO-Aware Water-Filling Scheduling

This fork adds a scheduling policy that serves each request "just enough" and spends the rest of the budget on requests that are falling behind. The rest of this README is upstream vLLM's.

### Idea

When the KV cache is over-subscribed, first-come-first-served leaves some requests fast and others starved. But clients do not consume output at the same speed: a person reads a chat reply at roughly 10 tokens/s, while an agent consumes everything at once. Tokens generated faster than they are read are wasted, so a request that is already ahead of its reader can wait while one that is behind gets the compute and the KV blocks.

Each request declares the output rate it needs, its SLO rate `r`, and the scheduler tracks how well it is met with a **satisfaction ratio**:

```text
s = generated_tokens / ((now - arrival_time) * r)
```

`s = 1` means the request has produced exactly what a reader at rate `r` would have consumed since it arrived. Time spent queued counts against the request. Water-filling then raises the lowest `s` first, like pouring water into the lowest vessel:

- **Running requests** are ordered by ascending `s`, so the token budget goes to the least satisfied.
- **Waiting requests** are admitted by ascending `s`.
- **On KV-cache pressure** the request with the largest `s`, the one furthest ahead, is preempted. Preemption recomputes, but the generated tokens are kept, so its `s` carries over.
- A request that declares no rate has `s = inf`: it is admitted last and preempted first.

### Design

- **Scheduling policy and victim policy are separate.** `--scheduling-policy` decides which waiting request is admitted next. `--preemption-victim-policy` decides which running request is evicted. When the victim policy is not set it is derived from the scheduling policy, so existing `fcfs` and `priority` behavior is unchanged, and the two can be combined freely.

  | `--scheduling-policy` | default `--preemption-victim-policy` |
  | --- | --- |
  | `fcfs` | `tail`: the most recently admitted request |
  | `priority` | `priority`: the lowest-priority request |
  | `waterfill` | `satisfaction`: the request with the largest `s` |

- **Output pacing is independent of scheduling.** A streaming request with a `target_rate` has its output buffered by the server and released in chunks at that rate, so the client sees tokens when it needs them, not when the engine produced them. The scheduler reads `target_rate` from the request itself and does not depend on pacing.
- **The queue ordering clock is pinned per step.** `s` changes with time, so the scheduler fixes one `now` per step for the waiting queues. Otherwise the request it peeks at could differ from the one it pops.

| Where | What |
| --- | --- |
| `vllm/config/scheduler.py` | `waterfill` policy, `preemption_victim_policy` |
| `vllm/v1/request.py` | `slo_rate`, `satisfaction_ratio()` |
| `vllm/v1/core/sched/request_queue.py` | `WaterfillRequestQueue` |
| `vllm/v1/core/sched/scheduler.py` | running order, `_select_preemption_victim()` |
| `vllm/v1/engine/rate_control.py`, `output_processor.py`, `async_llm.py` | paced delivery |

### Usage

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --scheduling-policy waterfill
```

Declare the SLO per request with `vllm_xargs`:

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true,
  "vllm_xargs": {"target_rate": 10, "chunk_size": 8}
}
```

| `vllm_xargs` | Meaning |
| --- | --- |
| `target_rate` | SLO output rate in tokens/s. Enables the satisfaction ratio and, for streaming deltas of one sequence (`n=1`), paced delivery. |
| `chunk_size` | Most tokens sent in one paced chunk. Default 16. |
| `pace` | `false` keeps the SLO for scheduling but delivers at full speed. |

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
vllm download-kernels  # precompiled kernels for faster startup
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
