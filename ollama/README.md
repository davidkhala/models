
`ollama run <model>` includes

1. `ollama pull <model>`

- typical sample models: `llama3.2`, `mistral`

2. `ollama serve`

`curl http://localhost:<port>/api/tags`

- for health check, expect return model list

[python sdk](https://github.com/ollama/ollama-python)

> Ollama does not require authentication for local API access
