from flask import Flask, request, jsonify

import os
from dotenv import load_dotenv

import json

import spacy

from firebase_admin import credentials, firestore, initialize_app

load_dotenv()

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Initialize Firestore DB
service_account = json.loads(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
cred = credentials.Certificate(service_account)
firebase_url = os.environ.get('FIREBASE_URL')
default_app = initialize_app(cred, options={"databaseURL": firebase_url})
db = firestore.client()
articles_ref = db.collection('articles')

app = Flask(__name__)


@app.route("/", methods=['POST'])
def process():
    data = request.get_json()
    article_id = data.get('id')

    article_snapshot = articles_ref.document(article_id).get()

    if not article_snapshot.exists:
        print('No such document!')
        return jsonify(data=None, error="No such document!"), 404

    article = article_snapshot.to_dict()

    if not article['parsed']:
        print('Article is not parsed!')
        return jsonify(data=None, error="Article is not parsed!"), 400

    if article['processed']:
        print('No action. Article has already been processed before!')
        return jsonify(data=article, error=None)

    print('Processing article...')

    header = nlp(article['content']['maybeHeader'])
    content = nlp(article['content']['mainContent'])

    header_without_stopwords = list(filter(lambda t: not t.is_stop, header))
    content_without_stopwords = list(filter(lambda t: not t.is_stop, content))

    header_data = list(map(lambda t: {
        "text": t.text,
        "lemma": t.lemma_,
        "pos": t.pos_,
    }, header_without_stopwords))

    content_data = list(map(lambda t: {
        "text": t.text,
        "lemma": t.lemma_,
        "pos": t.pos_,
    }, content_without_stopwords))

    # header_ents = list(map(lambda s: (s.text, s.lemma_), header.ents))

    # add other things: count, most common, keywords, toxic lang, etc.

    resp = {
        "header": header_data,
        "content": content_data
    }

    print('Saving to Firestore...')
    articles_ref.document(article_id).update({"nlp": resp, "processed": True})

    print('Article has been successfully processed!')
    return jsonify(data=resp, error=None)
