import os
import json
import traceback
import threading
import requests
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
)

class T5GenerationPipeline:
    def __init__(self, model_name_or_path: str, device: str) -> None:
        self.device = device
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config).to(self.device)

    def generate(self, input_text: str, max_input_length: int, max_output_length: int,  **kwargs) -> str:
        default_trunction_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        model_inputs = self.tokenizer([input_text], max_length=max_input_length, truncation=True, return_tensors="pt")
        self.tokenizer.truncation_side = default_trunction_side

        outputs = self.t5_model.generate(
            input_ids=model_inputs.input_ids.to(self.device),
            attention_mask=model_inputs.attention_mask.to(self.device),
            max_length=max_output_length,
            **kwargs
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text

def create_t5_generation_function(
        model_name_or_path: str, device: str, use_background_generation_server: bool
    ) -> callable:

    # Load T5 model
    t5_generation_pipeline = T5GenerationPipeline(model_name_or_path=model_name_or_path, device=device)

    # Option 1: Simply return a generation function of T5 generation pipeline
    if not use_background_generation_server:
        return t5_generation_pipeline.generate

    # Option 2: Run T5 generation API server in background and return a caller function
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # Avoid tokenizer warning
    class T5GenerationAPIHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                content_length = int(self.headers["Content-Length"])
                body = self.rfile.read(content_length)
                gen_kwargs = json.loads(body.decode("utf-8"))

                output_text = t5_generation_pipeline.generate(**gen_kwargs)
                response_body = {"output_text": output_text}
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()

                self.wfile.write(json.dumps(response_body).encode("utf-8"))
            
            except Exception as e:
                traceback.print_exc()
                response_body = {"error": str(e)}

                self.send_response(500)
                self.end_headers()

                self.wfile.write(json.dumps(response_body).encode("utf-8"))

    # Find an available port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    server_host, server_port = s.getsockname()
    s.close()

    def _run_server():
        try:
            server = HTTPServer((server_host, server_port), T5GenerationAPIHandler)
            server.serve_forever()
        except KeyboardInterrupt as e:
            server.socket.close()
            raise e
    
    server_thread = threading.Thread(target=_run_server)
    server_thread.start()
    print(f"T5 generation server started in background ({server_host}:{server_port})")

    def call_t5_generation_server(**kwargs) -> str:
        response = requests.post(
            url=f"http://{server_host}:{server_port}",
            json=kwargs
        )
        response_body = json.loads(response.content.decode("utf-8"))
        if "error" in response_body:
            raise RuntimeError(response_body["error"])
        return response_body["output_text"]

    return call_t5_generation_server
