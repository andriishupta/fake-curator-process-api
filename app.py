from flask import Flask, request, jsonify

from src.api.firebase import articles_ref
from src.api.nlp import nlp

from src.general import filter_doc, doc_summarization
from src.narrative import extract_keywords
from src.persuasion import detect_bigram_frequency, detect_lemma_frequency, detect_biased_language
from src.sentiment import analyze_sentiment

app = Flask(__name__)


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
    header = nlp(header_text)

    # create body Doc without stopwords
    body_text = article_data['content']['mainContent']
    body_full_doc = nlp(body_text)
    body_sents_count = len(list(body_full_doc.sents))
    body = filter_doc(body_full_doc)

    # -----------General-------------- #
    general_data = {
        "token_count": len(body),
        "sentences_count": body_sents_count,
    }

    # -----------Persuasion-------------- #
    # Repetition | Bias Language | Red Herring | Slogans
    lemma_frequency = detect_lemma_frequency(body)
    persuasion_data = {
        "lemma_frequency": lemma_frequency,
        "bigram_frequency": detect_bigram_frequency(body),
        **detect_biased_language(body),
    }

    # -----------Narrative-------------- #
    # Main idea | keywords | category topic(sport, politics) | entities | events
    summary = doc_summarization(body)
    summary_text = " ".join([sent.text for sent in summary])

    narrative_data = {
        "meta_keywords": [],  # get from parsing
        "keywords": extract_keywords(lemma_frequency),
        "header": header.text,
        "summary": summary_text,
        "idea_similarity_ratio": header.similarity(summary)
    }
    # header_ents = list(map(lambda s: (s.text, s.lemma_), header.ents))

    # -----------Toxicity-------------- #
    # todo: implement toxicity detection with Twitter introduction to the system.
    # Classifiers: toxic | severe_toxic | obscene | threat | insult | identity_hate
    # toxicity_data = detect_toxic_data(body)
    toxicity_data = {}

    # -----------Contextual Fact-Checking-------------- #
    # Detecting Context and process it with fact-checking API
    contextual_fact_checking_data = {}

    # -----------Sentiment Analysis-------------- #
    # Sentiment Analysis
    sentiment_analysis_data = analyze_sentiment(body)

    processed_data = {
        "general": general_data,
        "persuasion": persuasion_data,
        "narrative": narrative_data,
        "toxicity": toxicity_data,
        "contextual_fact_checking": contextual_fact_checking_data,
        "sentiment_analysis": sentiment_analysis_data,
    }

    print('Saving to Firestore...')
    articles_ref.document(article_id).update({
        "processed_data": processed_data,
        "processed": True
    })

    print('Article has been successfully processed!')
    return jsonify(data=processed_data, error=None)
