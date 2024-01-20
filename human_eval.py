import os
import json
import datetime
import argparse
import traceback

from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from utils.human_eval_ui_2cols import INTERFACE_HTML
from utils.human_eval_worlds import JMultiWOZWorld

TOD_MODEL_KWARGS = {
    "t5-base": {
        "model_name_or_path": "nu-dialogue/t5-base-jmultiwoz-e2e",
        "device": "cuda:0",
        "use_background_generation_server": True,
        "max_context_turns": 0,
        "max_input_length": 512,
        "max_output_length": 256,
        "dst_task_prefix": "対話から信念状態を推定:",
        "rg_task_prefix": "対話から応答を生成:",
        "user_utterance_prefix": "<顧客>",
        "system_utterance_prefix": "<店員>",
        "state_prefix": "<信念状態>",
        "db_result_prefix": "<検索結果>",
        "max_candidate_entities": 3,
        "book_result_prefix": "<予約結果>",
    },
    "t5-large": {
        "model_name_or_path": "tod_models/t5/output/t5-large-bs32-ep5-olen256/checkpoints",
        "device": "cuda:0",
        "use_background_generation_server": True,
        "max_context_turns": 0,
        "max_input_length": 512,
        "max_output_length": 256,
        "dst_task_prefix": "対話から信念状態を推定:",
        "rg_task_prefix": "対話から応答を生成:",
        "user_utterance_prefix": "<顧客>",
        "system_utterance_prefix": "<店員>",
        "state_prefix": "<信念状態>",
        "db_result_prefix": "<検索結果>",
        "max_candidate_entities": 3,
        "book_result_prefix": "<予約結果>",
    },
    "gpt3.5-fs": {
        "openai_model_name": "gpt-3.5-turbo",
        "max_context_turns": 5, # Use 5 context turns on OpenAI model
        "max_output_length": 256,
        "user_utterance_prefix": "<顧客>",
        "system_utterance_prefix": "<店員>",
        "state_prefix": "<信念状態>",
        "db_result_prefix": "<検索結果>",
        "max_candidate_entities": 3,
        "book_result_prefix": "<予約結果>",
        "faiss_db_fprefix": "tod_models/llm/output/faiss_db/hf-sup-simcse-ja-large-ctx2-d20",
        "num_fewshot_examples": 2,
    },
    "gpt4-fs": {
        "openai_model_name": "gpt-4",
        "max_context_turns": 5, # Use 5 context turns on OpenAI model
        "max_output_length": 256,
        "user_utterance_prefix": "<顧客>",
        "system_utterance_prefix": "<店員>",
        "state_prefix": "<信念状態>",
        "db_result_prefix": "<検索結果>",
        "max_candidate_entities": 3,
        "book_result_prefix": "<予約結果>",
        "faiss_db_fprefix": "tod_models/llm/output/faiss_db/hf-sup-simcse-ja-large-ctx2-d20",
        "num_fewshot_examples": 2,
    }
}

def tod_model_factory(tod_model_name: str):
    print(f"Loading {tod_model_name} ...")
    if tod_model_name == "t5-base":
        from tod_models.t5 import T5TODModel
        tod_model = T5TODModel(**TOD_MODEL_KWARGS["t5-base"])

    elif tod_model_name == "t5-large":
        from tod_models.t5 import T5TODModel
        tod_model = T5TODModel(**TOD_MODEL_KWARGS["t5-large"])
    
    elif tod_model_name == "gpt3.5-fs":
        from tod_models.llm import OpenAIFewShotTODModel
        tod_model = OpenAIFewShotTODModel(**TOD_MODEL_KWARGS["gpt3.5-fs"])
    
    elif tod_model_name == "gpt4-fs":
        from tod_models.llm import OpenAIFewShotTODModel
        tod_model = OpenAIFewShotTODModel(**TOD_MODEL_KWARGS["gpt4-fs"])
    
    else:
        raise ValueError(f"Unknown tod_model_name: {tod_model_name}")
    
    return tod_model


def run_server(args):
    # Save args
    os.makedirs(args.output_dpath) # Do not overwrite existing directory.

    json.dump(
        args.__dict__, open(os.path.join(args.output_dpath, "human_eval_args.json"), "w"),
        indent=4, ensure_ascii=False
    )
    json.dump(
        TOD_MODEL_KWARGS, open(os.path.join(args.output_dpath, "tod_model_args.json"), "w"),
        indent=4, ensure_ascii=False
    )
    
    # Load task IDs and session init params
    if args.task_ids_fpath is not None:
        task_ids = json.load(open(args.task_ids_fpath, "r"))
        json.dump(
            task_ids, open(os.path.join(args.output_dpath, "task_ids.json"), "w"),
            indent=4, ensure_ascii=False
        )
    else:
        task_ids = {}
    
    # Load TOD models
    tod_models = {}
    for tod_model_name in args.tod_model_names:
        tod_models[tod_model_name] = tod_model_factory(tod_model_name)

    # Create world for human evaluation
    jmultiwoz_world = JMultiWOZWorld(
        tod_models=tod_models,
        dataset_dpath=args.jmultiwoz_dataset_dpath,
        max_turns=args.max_turns,
    )

    class MyHTTPRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths = {
                '/': {'status': 200},
                '/dialogue': {'status': 200},
                '/favicon.ico': {'status': 202},  # Need for chrome
            }
            parsed_path = urlparse(self.path)
            if not parsed_path.path in paths.keys():
                response = 500
                self.send_response(response)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                content = ""
                self.wfile.write(bytes(content, 'UTF-8'))
                return
            
            response = paths[parsed_path.path]['status']

            print('headers\r\n-----\r\n{}-----'.format(self.headers))

            self.send_response(response)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()

            if parsed_path.path == '/':
                response = 500
                self.send_response(response)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                content = ""
                self.wfile.write(bytes(content, 'UTF-8'))
                return

            if parsed_path.path == '/dialogue':
                sess_init_params = {}

                if task_ids:
                    query = parse_qs(parsed_path.query)
                    try:
                        sess_init_params = task_ids[query["task_id"][0]]
                    except Exception as e:
                        content = ( "<html><body>"
                                   f"<h3>Error: {e}</h3>"
                                   f"<p>Please specify a valid <b>task_id</b> in the URL query.</p>"
                                    "</body></html>")
                        response = 500
                        self.send_response(response)
                        self.send_header('Content-Type', 'text/html; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(bytes(content, 'UTF-8'))
                        return

                html_format_args = jmultiwoz_world.create_new_session(**sess_init_params)
                content = INTERFACE_HTML.format(
                    instruction=html_format_args["instruction"],
                    session_id=html_format_args["session_id"],
                    question_list=html_format_args["question_list"],
                    answer_list=html_format_args["answer_list"],
                )
                self.wfile.write(bytes(content, 'UTF-8'))


        def do_POST(self):
            """
            Handle POST request, especially replying to a chat message.
            """
            print('path = {}'.format(self.path))
            parsed_path = urlparse(self.path)
            print('parsed: path = {}, query = {}'.format(parsed_path.path, parse_qs(parsed_path.query)))

            print('headers\r\n-----\r\n{}-----'.format(self.headers))

            if self.path == '/interact':
                content_length = int(self.headers['content-length'])
                try:
                    content = self.rfile.read(content_length).decode('utf-8')
                    body = json.loads(content)

                    print('body = {}'.format(body))

                    model_utterance, session_over = jmultiwoz_world.model_response(
                        session_id=body["sessionId"],
                        user_input=body["userInput"],
                    )

                    if model_utterance is not None:
                        print("sys: " + model_utterance, flush=True)
                    model_response = {"text": model_utterance, "sessionOver": session_over}

                except Exception as e:
                    print("error", traceback.format_exc(), flush=True)
                    model_response = {"text": f"error Message: {e}", "session_over": False}
                    # raise e

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                json_str = json.dumps(model_response)
                self.wfile.write(bytes(json_str, 'utf-8'))

            elif self.path == '/evaluate':
                content_length = int(self.headers['content-length'])
                try:
                    content = self.rfile.read(content_length).decode('utf-8')
                    body = json.loads(content)

                    print('body = {}'.format(body))

                    print(f"Saving evaluation: {body['sessionId']}", flush=True)
                    jmultiwoz_world.save_eval_scores(
                        session_id=body["sessionId"],
                        eval_scores=body["evalValues"],
                    )
                    
                    print(f"Terminating session: {body['sessionId']}", flush=True)
                    jmultiwoz_world.export_session(
                        session_id=body["sessionId"],
                        sessions_dpath=os.path.join(args.output_dpath, "sessions"),
                    )
                    jmultiwoz_world.terminate_session(session_id=body["sessionId"])

                    model_response = {"text": "Received evaluation results and terminated session."}
                
                except Exception as e:
                    # print("error", e, flush=True)
                    # model_response = {"text": f"error Message: {e}"}
                    raise e
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                json_str = json.dumps(model_response)
                self.wfile.write(bytes(json_str, 'utf-8'))

    if args.threading_httpd:
        print("*** Use ThreadingHTTPServer ***", flush=True)
        http_server_class = ThreadingHTTPServer
    else:
        print("*** Use HTTPServer ***", flush=True)
        http_server_class = HTTPServer

    print("Start", flush=True)
    try:
        MyHTTPRequestHandler.protocol_version = 'HTTP/1.0'
        with http_server_class(('localhost', 8080), MyHTTPRequestHandler) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        jmultiwoz_world.export_unterminated_sessions(
            sessions_dpath=os.path.join(args.output_dpath, "unterminated_sessions"),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tod_model_names", type=str, nargs="+", required=True,
                        help="List of TOD model names.")
    parser.add_argument("--task_ids_fpath", type=str, default=None,
                        help="Path to file listing task IDs and the session init params for each task.")
    parser.add_argument("--output_dpath", type=str,
                        default=f"human_eval_output/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        help="Path to directory to save results.")
    parser.add_argument("--jmultiwoz_dataset_dpath", type=str, default="dataset/JMultiWOZ_1.0",
                        help="Path to JMultiWOZ dataset.")
    parser.add_argument("--max_turns", type=int, default=40,
                        help="Max turn per dialogue session.")
    parser.add_argument("--threading_httpd", action="store_true",
                        help="Use ThreadingHTTPServer instead of HTTPServer.")
    args = parser.parse_args()

    run_server(args)
