from http.server import BaseHTTPRequestHandler
from urllib import parse

import spacy

nlp = spacy.load("en_core_web_sm")


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

        for token in doc:
            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha,
                  token.is_stop)

        s = self.path
        dic = dict(parse.parse_qsl(parse.urlsplit(s).query))
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

        if "name" in dic:
            message = "Hello, " + dic["name"] + "!"
        else:
            message = "Hello, stranger!"

        self.wfile.write(message.encode())
        return
