from flask import Flask, request, jsonify

import os

from src.api.firebase import articles_ref

from src.general import clean_doc, summarize_doc, top_bigram_frequency, top_lemma_frequency
from src.narrative import extract_keywords
from src.persuasion import detect_biased_language, detect_paraphrased_ideas, detect_dehumanizing_language
from src.sentiment import analyze_sentiment

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

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

    if 'parsed_data' not in article_data:
        print('Article is not parsed!')
        return jsonify(data=None, error="Article is not parsed!"), 400

    if 'processed_data' in article_data:
        print('No action. Article has already been processed before!')
        return jsonify(data=article_data, error=None)

    print('Processing article...')

    # todo parse <meta/> for keywords and other useful info
    # article['processed_data']['meta']['title' | 'description']

    header_text = article_data['parsed_data']['header']
    header_doc = clean_doc(header_text)

    body_text = article_data['parsed_data']['body']
    body_doc = clean_doc(body_text)

    print("# -----------General-------------- #")
    lemma_frequency = top_lemma_frequency(body_doc)
    bigram_frequency = top_bigram_frequency(body_doc)
    general_data = {
        "token_count": len(body_doc),
        "sentences_count": len(list(body_doc.sents)),
        "lemma_frequency": lemma_frequency,
        "bigram_frequency": bigram_frequency,
    }

    print("# -----------Persuasion(Influencing beliefs)-------------- #")
    # Paraphrase(Repetition) | Bias Language | TODO: Red Herring(distraction) | TODO: Slogans
    persuasion_data = {
        "paraphrased_ratio": detect_paraphrased_ideas(body_doc),
        "dehumanizing_language_ratio": detect_dehumanizing_language(body_doc),
        **detect_biased_language(body_doc)
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
        })
        print('Article has been successfully processed!')

    return jsonify(data=processed_data, error=None)
