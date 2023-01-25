import spacy
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello_world():
    doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha,
              token.is_stop)

    return "<p>Hello, World!</p>"