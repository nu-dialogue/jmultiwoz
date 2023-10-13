import json

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from interface import INTERFACE_HTML
from sessions import DialogueSessions

STYLE_SHEET = "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.4/css/bulma.css"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.3.1/js/all.js"
MAX_TURN = 20


def server(args):

    
    dialogue_sessions = DialogueSessions(
        tod_models=,
        dataset_dpath=...,
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
                    session_id, goal_description_str = dialogue_sessions.make_new_session()
                    content = INTERFACE_HTML.format(
                        style_sheet_href=STYLE_SHEET,
                        font_src=FONT_AWESOME,
                        session_id=session_id,
                        goal_description=goal_description_str,
                        max_turn=MAX_TURN,
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
                    
                    resp_text = dialogue_sessions.response(
                        session_id=body['session_id'],
                        input_text=body['user_input'],
                    )

                    print("sys: " + resp_text, flush=True)
                    model_response = {"text": resp_text}
                except Exception as e:
                    print("error", e, flush=True)
                    model_response = {"text": f"error Message: {e}"}

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                json_str = json.dumps(model_response)
                self.wfile.write(bytes(json_str, 'utf-8'))


    print("Start", flush=True)
    address = ('localhost', 8080)

    MyHTTPRequestHandler.protocol_version = 'HTTP/1.0'
    with HTTPServer(address, MyHTTPRequestHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    server()