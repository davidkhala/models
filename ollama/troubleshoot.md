
# 模型加载
通过日志判断
- Ollama 目前没有任何 API 能告诉你 GPU offload 状态。

- tail日志: `ollama serve`
  - `tail -f ~/.ollama/logs/server.log`

GPU usage hints
- KV Cache on GPU: `llama_kv_cache:      CUDA0 KV buffer size =   512.00 MiB`
- Compute buffer on GPU: `sched_reserve:      CUDA0 compute buffer size =   112.01 MiB`
- Flash Attention enabled: `sched_reserve: Flash Attention was auto, set to enabled`

