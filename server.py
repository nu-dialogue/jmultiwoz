import os
import json
import datetime
import argparse

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
# from interface import INTERFACE_HTML
from interface_2cols import INTERFACE_HTML
from world import TOD_MODEL_KWARGS, JMultiWOZWorld

STYLE_SHEET = "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.4/css/bulma.css"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.3.1/js/all.js"

def server(args):
    os.makedirs(args.output_dpath) # Do not overwrite existing directory.

    json.dump(
        args.__dict__, open(os.path.join(args.output_dpath, "human_eval_args.json"), "w"),
        indent=4, ensure_ascii=False
    )
    json.dump(
        TOD_MODEL_KWARGS, open(os.path.join(args.output_dpath, "tod_model_args.json"), "w"),
        indent=4, ensure_ascii=False
    )
    
    jmultiwoz_world = JMultiWOZWorld(
        tod_model_names=args.tod_model_names,
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
            if not urlparse(self.path).path in paths.keys():
                response = 500
                self.send_response(response)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                content = ""
                self.wfile.write(bytes(content, 'UTF-8'))
            else:
                parsed_path = urlparse(self.path)
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

                elif parsed_path.path == '/dialogue':
                    html_format_args = jmultiwoz_world.create_new_session()
                    content = INTERFACE_HTML.format(
                        stylesheet_href=STYLE_SHEET,
                        font_src=FONT_AWESOME,
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
                    print("error", e, flush=True)
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


    print("Start", flush=True)
    address = ('localhost', 8080)

    try:
        MyHTTPRequestHandler.protocol_version = 'HTTP/1.0'
        with HTTPServer(address, MyHTTPRequestHandler) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        jmultiwoz_world.export_unterminated_sessions(
            sessions_dpath=os.path.join(args.output_dpath, "unterminated_sessions"),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tod_model_names", type=str, nargs="+", required=True,
                        help="List of TOD model names.")
    parser.add_argument("--output_dpath", type=str,
                        default=f"human_eval_output/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        help="Path to directory to save results.")
    parser.add_argument("--jmultiwoz_dataset_dpath", type=str, default="dataset/JMultiWOZ_1.0",
                        help="Path to JMultiWOZ dataset.")
    parser.add_argument("--max_turns", type=int, default=40,
                        help="Max turn per dialogue session.")
    args = parser.parse_args()

    server(args)
