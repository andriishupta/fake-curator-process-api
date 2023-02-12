from flask import Flask, request, jsonify

import spacy

from src.general import filter_stopwords

from src.api.firebase import articles_ref

app = Flask(__name__)

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


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

    # create header Doc as is
    header_text = article['content']['maybeHeader']  # todo change content to raw_data
    header = nlp(header_text)

    # create body Doc without stopwords
    body_text = article['content']['mainContent']
    body_full_doc = nlp(body_text)
    body_clean = filter_stopwords(body_full_doc)
    body_clean_text = " ".join([token.text for token in body_clean])
    body = nlp(body_clean_text)

    # -----------General-------------- #
    general_data = {
        "words_count": 10,
        "sentences_count": 2,
    }

    # -----------Narrative-------------- #
    # Main idea | keywords | category topic(sport, politics) | entities | events
    narrative_data = {
        "theme": '',
    }
    # header_ents = list(map(lambda s: (s.text, s.lemma_), header.ents))

    # -----------Toxicity-------------- #
    # todo: implement toxicity detection with Twitter introduction to the system.
    # Classifiers: toxic | severe_toxic | obscene | threat | insult | identity_hate
    # toxicity_data = detect_toxic_data(body)
    toxicity_data = {}

    # -----------Persuasion-------------- #
    # Repetition | Loaded language | Bias Language | Red Herring | Slogans
    # top_lemma_frequency
    persuasion_data = {}

    # -----------Contextual Fact-Checking-------------- #
    # Detecting Context from narrative and process it with different API
    contextual_fact_checking_data = {}

    # -----------Sentiment Analysis-------------- #
    # Sentiment Analysis
    sentiment_analysis_data = {}

    processed_data = {
        "general": general_data,
        "narrative": narrative_data,
        "toxicity": toxicity_data,
        "persuasion": persuasion_data,
        "contextual_fact_checking": contextual_fact_checking_data,
        "sentiment_analysis": {},
    }

    print('Saving to Firestore...')
    articles_ref.document(article_id).update({"procesed_data": processed_data, "processed": True})

    print('Article has been successfully processed!')
    return jsonify(data=processed_data, error=None)
