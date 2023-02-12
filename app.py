from flask import Flask, request, jsonify

import os

from src.api.firebase import articles_ref
from src.api.nlp import nlp

from src.general import filter_doc, summarize_doc, top_bigram_frequency, top_lemma_frequency
from src.narrative import extract_keywords
from src.persuasion import detect_biased_language, detect_paraphrased_ideas
from src.sentiment import analyze_sentiment

app = Flask(__name__)

is_production = os.environ.get("FLASK_ENV") == "production"


@app.route("/", methods=['POST'])
def process():
    data = request.get_json()
    article_id = data.get('id')

    article_snapshot = articles_ref.document(article_id).get()

    if not article_snapshot.exists:
        print('No such document!')
        return jsonify(data=None, error="No such document!"), 404

    article_data = article_snapshot.to_dict()

    if not article_data['parsed']:
        print('Article is not parsed!')
        return jsonify(data=None, error="Article is not parsed!"), 400

    if article_data['processed']:
        print('No action. Article has already been processed before!')
        return jsonify(data=article_data, error=None)

    print('Processing article...')

    # article['raw_data']['meta']  # todo parse <meta/> for keywords and other useful info
    # create header Doc as is
    header_text = article_data['content']['maybeHeader']  # todo change content to raw_data
    header_doc = nlp(header_text)

    # create body Doc without stopwords
    body_text = article_data['content']['mainContent']
    body_full_doc = nlp(body_text)
    body_sents_count = len(list(body_full_doc.sents))
    body_doc = filter_doc(body_full_doc)

    print("# -----------General-------------- #")
    lemma_frequency = top_lemma_frequency(body_doc)
    bigram_frequency = top_bigram_frequency(body_doc)
    general_data = {
        "token_count": len(body_doc),
        "sentences_count": body_sents_count,
        "lemma_frequency": lemma_frequency,
        "bigram_frequency": bigram_frequency,
    }

    print("# -----------Persuasion(Influencing beliefs)-------------- #")
    # Paraphrase(Repetition) | Bias Language | TODO: Red Herring(distraction) | TODO: Slogans
    persuasion_data = {
        **detect_biased_language(body_doc),
        "paraphrased_ratio": detect_paraphrased_ideas(body_doc),
    }

    print("# -----------Narrative-------------- #")
    # Main idea | keywords | entities
    summary_doc = summarize_doc(body_doc)
    summary_text = " ".join([sent.text for sent in summary_doc])

    narrative_data = {
        "keywords_meta": [],  # get from parsing <meta/>
        "keywords": extract_keywords(lemma_frequency),
        "keywords_bigrams": extract_keywords(bigram_frequency),
        "header": header_doc.text,
        "summary": summary_text,
        "header_summary_similarity_ratio": header_doc.similarity(summary_doc)
    }

    print("# -----------Toxicity-------------- #")
    # todo: implement toxicity detection with Twitter introduction to the system.
    # Classifiers: toxic | severe_toxic | obscene | threat | insult | identity_hate
    # toxicity_data = detect_toxic_data(body)
    toxicity_data = {}

    print("# -----------Contextual Fact-Checking-------------- #")
    # Detecting Context and process it with fact-checking API
    contextual_fact_checking_data = {}

    print("# -----------Sentiment Analysis-------------- #")
    # Sentiment Analysis
    sentiment_analysis_data = analyze_sentiment(body_doc)

    processed_data = {
        "general": general_data,
        "persuasion": persuasion_data,
        "narrative": narrative_data,
        "toxicity": toxicity_data,
        "contextual_fact_checking": contextual_fact_checking_data,
        "sentiment_analysis": sentiment_analysis_data,
    }

    if is_production:
        print('Saving to Firestore...')
        articles_ref.document(article_id).update({
            "processed_data": processed_data,
            "processed": True
        })
        print('Article has been successfully processed!')

    return jsonify(data=processed_data, error=None)
