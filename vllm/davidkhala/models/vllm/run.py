from vllm.entrypoints.openai.api_server import run_server

def server(model:str, host:str="0.0.0.0", port:int=8000):
    return run_server(
        model=model,
        host=host,
        port=port,
    )



