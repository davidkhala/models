from argparse import Namespace

from vllm.entrypoints.openai.api_server import run_server


async def server(model: str, host: str = "0.0.0.0", port: int = 8000):
    await run_server(Namespace(
        model=model,
        host=host,
        port=port,
    ))
