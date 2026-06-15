

`curl http://localhost:11434/api/tags`

- for health check, expect return model list

[python sdk](https://github.com/ollama/ollama-python)

> Ollama does not require authentication for local API access

## manual
`ollama run <model>`
- it opens interactive session only
- By default, ollama 默认会把模型保留大约 5 分钟（300 秒）after session closed
  - To customize life time, you need: `OLLAMA_KEEP_ALIVE=24h ollama run <model>`

`ollama list`: List models

`ollama ps`: List running models

`ollama serve`: attach to the daemon log