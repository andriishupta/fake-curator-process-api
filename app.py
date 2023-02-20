from flask import Flask, request, jsonify

import os
from itertools import islice
import csv
import json
import argparse

import numpy as np

from sklearn.manifold import MDS
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from src.api.firebase import articles_ref

from src.general import clean_doc, summarize_doc, top_bigram_frequency, top_lemma_frequency
from src.narrative import extract_keywords
from src.persuasion import detect_biased_language, detect_paraphrased_ideas, detect_dehumanizing_language
from src.sentiment import analyze_sentiment
from src.linguistic import detect_unusual_inappropriate_language_ratio, detect_awkward_text_ratio
from src.ner import get_ner_frequency

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

is_production = os.environ.get("FLASK_ENV") == "production"

CSV_LINES = 10


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
    body_text = article_data['parsed_data']['body']

    processed_data = process_text(header_text, body_text)

    if is_production:
        print('Saving to Firestore...')
        articles_ref.document(article_id).update({
            "processed_data": processed_data,
        })
        print('Article has been successfully processed!')

    return jsonify(data=processed_data, error=None)


def process_data():
    process_csv()
    process_mds()


def process_text(header_text, body_text):
    header_doc = clean_doc(header_text)
    body_doc = clean_doc(body_text)

    print("# -----------General-------------- #")
    lemma_frequency = top_lemma_frequency(body_doc)
    bigram_frequency = top_bigram_frequency(body_doc)
    general_data = {
        "token_count": len(body_doc),
        "sentences_count": len(list(body_doc.sents)),
        "lemma_frequency": lemma_frequency,
        "bigram_frequency": bigram_frequency,
        "ner_frequency": get_ner_frequency(body_doc),
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

    print("# -----------Linguistic Analysis-------------- #")
    linguistic_data = {
        "unusual_inappropriate_language_ratio": detect_unusual_inappropriate_language_ratio(body_doc),
        "awkward_text_ratio": detect_awkward_text_ratio(body_doc),
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
        "linguistic": linguistic_data,
        "toxicity": toxicity_data,
        "contextual_fact_checking": contextual_fact_checking_data,
        "sentiment_analysis": sentiment_analysis_data,
    }

    return processed_data


def process_csv():
    fake_data = []
    true_data = []

    with open('data/Fake.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in islice(enumerate(reader), CSV_LINES):
            print('processing Fake #', i)
            title = row['title']
            text = row['text']
            try:
                processed_data = process_text(title, text)
                fake_data.append(processed_data)
            except BaseException as e:
                print('error processing row #', i, ' msg: ', str(e))
                continue

    with open('data/True.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in islice(enumerate(reader), CSV_LINES):
            print('processing True #', i)
            title = row['title']
            text = row['text']
            try:
                processed_data = process_text(title, text)
                true_data.append(processed_data)
            except BaseException as e:
                print('error processing row #', i, ' msg: ', str(e))
                continue

    with open('out/fake.json', 'w') as f:
        json.dump(fake_data, f)

    with open('out/true.json', 'w') as f:
        json.dump(true_data, f)


def process_mds():
    # Load processed data from fake.json and true.json
    with open('out/fake.json', 'r') as f:
        fake_data = json.load(f)
    with open('out/true.json', 'r') as f:
        true_data = json.load(f)

    fake_values = []
    for item in fake_data:
        item_values = [
            item['persuasion']['paraphrased_ratio'],
            item['persuasion']['dehumanizing_language_ratio'],
            item['persuasion']['subjective_words_ratio'],

            item['narrative']['header_summary_similarity_ratio'],

            item['linguistic']['unusual_inappropriate_language_ratio'],
            item['linguistic']['awkward_text_ratio'],

            item['sentiment_analysis']['avg_sentiment'],
            item['sentiment_analysis']['positive_ratio'],
            item['sentiment_analysis']['neutral_ratio'],
            item['sentiment_analysis']['negative_ratio'],
        ]
        fake_values.append(item_values)

    mds = MDS(n_components=2).fit_transform(np.array(fake_values))
    fake_mds = mds.tolist()

    true_values = []
    for item in true_data:
        item_values = [
            item['persuasion']['paraphrased_ratio'],
            item['persuasion']['dehumanizing_language_ratio'],
            item['persuasion']['subjective_words_ratio'],

            item['narrative']['header_summary_similarity_ratio'],

            item['linguistic']['unusual_inappropriate_language_ratio'],
            item['linguistic']['awkward_text_ratio'],

            item['sentiment_analysis']['avg_sentiment'],
            item['sentiment_analysis']['positive_ratio'],
            item['sentiment_analysis']['neutral_ratio'],
            item['sentiment_analysis']['negative_ratio'],
        ]
        true_values.append(item_values)

    mds = MDS(n_components=2).fit_transform(np.array(true_values))
    true_mds = mds.tolist()

    # Save output to fake_mds.json and true_mds.json
    with open('out/fake_mds.json', 'w') as f:
        json.dump(fake_mds, f)
    with open('out/true_mds.json', 'w') as f:
        json.dump(true_mds, f)


def plot_results():
    with open('out/fake_mds.json', 'r') as f:
        fake_mds = json.load(f)
    with open('out/true_mds.json', 'r') as f:
        true_mds = json.load(f)

    # Plot true MDS values in green and fake MDS values in red
    plt.scatter([p[0] for p in true_mds], [p[1] for p in true_mds], c="g")
    plt.scatter([p[0] for p in fake_mds], [p[1] for p in fake_mds], c="r")

    # Add axis labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Fake/Real News Visualization")

    # Show the plot
    plt.show()


def plot_kmeans():
    with open('out/fake_mds.json', 'r') as f:
        fake_mds = json.load(f)
    with open('out/true_mds.json', 'r') as f:
        true_mds = json.load(f)

    # Combine the MDS points into a single dataset
    data = np.concatenate((true_mds, fake_mds), axis=0)

    # Set the number of clusters (you can experiment with different values)
    n_clusters = 2

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

    # Get the cluster labels for each data point
    labels = kmeans.labels_

    # Visualize the clustering
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3,
                color='r')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function_name', help='Name of function to run')
    args = parser.parse_args()

    if args.function_name == 'process_data':
        process_data()
    elif args.function_name == 'plot_results':
        plot_results()
    elif args.function_name == 'plot_kmeans':
        plot_kmeans()
    else:
        print(f'Error: Invalid function name "{args.function_name}"')
