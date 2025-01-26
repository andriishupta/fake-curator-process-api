from itertools import islice
import csv
import json
import argparse
import random

import numpy as np
import pandas as pd

from sklearn.manifold import MDS
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

from src.general import clean_doc, summarize_text, top_bigram_frequency, top_lemma_frequency, extend_text_with_explanation
from src.persuasion import detect_biased_language, detect_paraphrased_ideas, detect_dehumanizing_language_ratio
from src.sentiment import analyze_sentiment
from src.linguistic import detect_unusual_inappropriate_language_ratio, detect_awkward_text_ratio
from src.ner import get_ner_frequency

CSV_LINES = 1000

def process_data():
    process_csv()
    process_mds()


def process_csv():
    lemmas = set()
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
                lemmas.update(processed_data["general"]["top_lemmas"])
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
                lemmas.update(processed_data["general"]["top_lemmas"])
                true_data.append(processed_data)
            except BaseException as e:
                print('error processing row #', i, ' msg: ', str(e))
                continue

    with open('out/fake.json', 'w') as f:
        json.dump(fake_data, f)

    with open('out/true.json', 'w') as f:
        json.dump(true_data, f)

    with open('out/lemmas.json', 'w') as f:
        json.dump(list(lemmas), f)


def process_text(header_text, body_text):
    header_doc = clean_doc(header_text)
    body_doc = clean_doc(body_text)

    print("# -----------General-------------- #")
    lemma_frequency, top_lemmas = top_lemma_frequency(body_doc)
    bigram_frequency = top_bigram_frequency(body_doc)
    general_data = {
        "token_count": len(body_doc),
        "sentences_count": len(list(body_doc.sents)),
        "lemma_frequency": lemma_frequency,
        "top_lemmas": top_lemmas,
        "bigram_frequency": bigram_frequency,
        "ner_frequency": get_ner_frequency(body_doc),
    }

    print("# -----------Persuasion(Influencing beliefs)-------------- #")
    # Paraphrase(Repetition) | Bias Language | TODO: Red Herring(distraction) | TODO: Slogans
    persuasion_data = {
        "paraphrased_ratio": detect_paraphrased_ideas(body_doc),
        "dehumanizing_language_ratio": detect_dehumanizing_language_ratio(body_doc),
        **detect_biased_language(body_doc)
    }

    print("# -----------Narrative-------------- #")
    # Main idea | keywords | entities
    summary_text, summary_doc = summarize_text(body_doc)

    narrative_data = {
        "keywords_meta": [],  # get from parsing <meta/>
        "keywords": list(lemma_frequency),
        "keywords_bigrams": list(bigram_frequency),
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
    # toxicity_classifiers = detect_toxic_data(body_doc)
    # print(toxicity_classifiers)
    # toxicity_data = {
    #     "classifiers": toxicity_classifiers,
    #     "avg_mark": detect_toxicity_ratio(toxicity_classifiers)
    # }

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
        # "toxicity": toxicity_data,
        # "contextual_fact_checking": contextual_fact_checking_data,
        "sentiment_analysis": sentiment_analysis_data,
    }

    print("# -----------Post Process-------------- #")
    processed_data["extended_text_with_explanation"] = extend_text_with_explanation(body_text, processed_data)

    return processed_data


def process_mds():
    # Load processed data from fake.json and true.json
    with open('out/fake.json', 'r') as f:
        fake_data = json.load(f)
    with open('out/true.json', 'r') as f:
        true_data = json.load(f)
    with open('out/lemmas.json', 'r') as f:
        lemmas = json.load(f)

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

            *lemmas_params(lemmas, item['general']['lemma_frequency'])
        ]
        fake_values.append(item_values)

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

            *lemmas_params(lemmas, item['general']['lemma_frequency'])
        ]
        true_values.append(item_values)

    mds = MDS().fit_transform(np.array(true_values + fake_values))
    # mds = Isomap.fit_transform(np,true_values + fake_values)

    # Save output to fake_mds.json and true_mds.json
    with open('out/mds_results.json', 'w') as f:
        json.dump({
            "mds": mds.tolist(),
            "true": len(true_values),
            "fake": len(fake_values),
        }, f)

def lemmas_params(all_lemmas, item_lemmas_dict):
    params = []
    for lemma in all_lemmas:
        if lemma in item_lemmas_dict:
            params.append(item_lemmas_dict[lemma])
        else:
            params.append(0)
    return params

def plot_results():
    with open('out/mds_results.json', 'r') as f:
        mds_results = json.load(f)

    X = np.array(mds_results["mds"])
    y = np.concatenate((np.ones(mds_results["true"]), np.zeros(mds_results["fake"])))

    colors = np.where(y == 1, 'g', 'r')

    # Plot true MDS values in green and fake MDS values in red
    plt.scatter(X[:, 0], X[:, 1], c=colors)

    # Add axis labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Fake/Real News Visualization")

    # Show the plot
    plt.show()


def plot_clusters():
    with open('out/mds_results.json', 'r') as f:
        mds_results = json.load(f)

    X = np.array(mds_results["mds"])
    y = np.concatenate((np.ones(mds_results["true"]), np.zeros(mds_results["fake"])))

    # Perform k-means clustering
    k = 2  # Number of clusters
    kmeans = KMeans(n_clusters=k).fit(X)
    labels = kmeans.labels_

    # Generate scatter plot of MDS data
    fig, ax = plt.subplots()
    for label in np.unique(labels):
        mask = labels == label
        if label == 1:
            color = 'green'
        else:
            color = 'red'
        ax.scatter(X[mask, 0], X[mask, 1], c=color, label=label)

    # Highlight misclassified points in each cluster
    for label in np.unique(labels):
        mask = labels == label
        if label == 0:
            incorrect_label = 1  # True news in fake news cluster
        else:
            incorrect_label = 0  # Fake news in true news cluster
        opposite_mask = y[mask] == np.full(np.sum(mask), ('fake', 'real')[incorrect_label])
        opposite_count = np.sum(opposite_mask)
        if opposite_count > 0:
            opposite_color = 'red' if incorrect_label == 0 else 'green'
            ax.scatter(X[mask][opposite_mask, 0], X[mask][opposite_mask, 1], c=opposite_color, marker='x')

    # Add plot title and legend
    ax.set_title('MDS plot with k-means clustering')
    ax.legend()

    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='k')
    plt.show()


# import tweepy
# from textblob import TextBlob
#
#
# # Function to analyze sentiment bias in Twitter comments
# def analyze_twitter_bias(tweet_id, api_key, api_secret, access_token, access_secret):
#     """
#     Analyzes the average bias of a Twitter conversation based on comments' sentiment.
#
#     Parameters:
#         tweet_id (str): The ID of the tweet to analyze.
#         api_key (str): Twitter API key.
#         api_secret (str): Twitter API secret key.
#         access_token (str): Twitter access token.
#         access_secret (str): Twitter access token secret.
#
#     Returns:
#         dict: A dictionary containing the average sentiment score and total comments analyzed.
#     """
#     # Authenticate with the Twitter API
#     auth = tweepy.OAuthHandler(api_key, api_secret)
#     auth.set_access_token(access_token, access_secret)
#     api = tweepy.API(auth)
#
#     try:
#         # Fetch replies to the tweet
#         replies = []
#         for tweet in tweepy.Cursor(api.search_tweets, q=f'to:{api.get_status(tweet_id).user.screen_name}',
#                                    since_id=tweet_id, tweet_mode='extended').items():
#             if hasattr(tweet, 'in_reply_to_status_id_str'):
#                 if tweet.in_reply_to_status_id_str == tweet_id:
#                     replies.append(tweet.full_text)
#
#         if not replies:
#             return {"average_sentiment": None, "total_comments": 0}
#
#         # Analyze sentiment of each reply
#         sentiment_scores = []
#         for reply in replies:
#             sentiment = TextBlob(reply).sentiment.polarity
#             sentiment_scores.append(sentiment)
#
#         # Calculate average sentiment
#         average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
#         return {
#             "average_sentiment": average_sentiment,
#             "total_comments": len(replies)
#         }
#
#     except Exception as e:
#         return {"error": str(e)}


# process only using TD-IDF extraction
def process_svm_default():
    true = pd.read_csv("data/True.csv")
    true['label'] = 1
    fake = pd.read_csv("data/Fake.csv")
    fake['label'] = 0

    df = pd.concat([true.head(CSV_LINES), fake.head(CSV_LINES)])
    X = df['text'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # extract features using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # train the SVM model
    svm = SVC().fit(X_train_tfidf, y_train)

    # evaluate the model on the testing set
    y_pred = svm.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("[Default] Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}".format(
        accuracy, precision, recall, f1))

    # Additional testing on randomly selected real news
    true_random = true.sample(n=100, random_state=42)
    fake_random = fake.sample(n=100, random_state=42)
    df_random = pd.concat([true_random, fake_random])
    X_random = vectorizer.transform(df_random['text'].values)
    y_random = df_random['label'].values

    y_random_pred = svm.predict(X_random)
    accuracy_random = accuracy_score(y_random, y_random_pred)
    precision_random = precision_score(y_random, y_random_pred)
    recall_random = recall_score(y_random, y_random_pred)
    f1_random = f1_score(y_random, y_random_pred)
    print("[Default][Random Values Testing] Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}".format(
        accuracy_random, precision_random, recall_random, f1_random))


def process_svm_feature_vectors():
    # 1) Load MDS results
    with open('out/mds_results.json', 'r') as f:
        mds_results = json.load(f)

    # 2) Build feature matrix X and labels y
    X = np.array(mds_results["mds"])
    # '1' for True articles, '0' for Fake articles
    y = np.concatenate((
        np.ones(mds_results["true"]),
        np.zeros(mds_results["fake"])
    ))

    # 3) Standard train/test split (all data)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),  # Keep track of original indices as well
        test_size=0.2,
        random_state=42
    )

    # 4) Train an SVM model
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)

    # 5) Evaluate on the main test split
    y_pred = svm.predict(X_test)
    accuracy_main = accuracy_score(y_test, y_pred)
    precision_main = precision_score(y_test, y_pred)
    recall_main = recall_score(y_test, y_pred)
    f1_main = f1_score(y_test, y_pred)

    # SINGLE-LINE PRINT for main train/test metrics
    print("[Feature Vectors] Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}"
          .format(accuracy_main, precision_main, recall_main, f1_main))

    # 6) Additional testing on a RANDOM SUBSET WITHIN THE TEST SET
    #    This guarantees no overlap with the training set.

    # figure out which test-set rows are "true" vs "fake"
    # (We know that if idx < mds_results["true"], it was originally True -> label=1, else label=0)
    total_true = mds_results["true"]
    # We'll look at the 'idx_test' array to see which original indices ended up in the test set
    test_true_indices = []
    test_fake_indices = []
    for i, original_idx in enumerate(idx_test):
        if original_idx < total_true:
            test_true_indices.append(i)  # 'i' is the local index in X_test
        else:
            test_fake_indices.append(i)

    # how many random "true" to test
    n_samples_true = 20
    # how many random "fake" to test
    n_samples_fake = 20

    # randomly sample local indices from test_true_indices / test_fake_indices
    random_true_indices = random.sample(
        test_true_indices,
        min(n_samples_true, len(test_true_indices))
    )
    random_fake_indices = random.sample(
        test_fake_indices,
        min(n_samples_fake, len(test_fake_indices))
    )

    # combine them
    random_subset_indices = random_true_indices + random_fake_indices

    # build X_random, y_random
    X_random = X_test[random_subset_indices]
    y_random = y_test[random_subset_indices]

    # Evaluate on random subset
    y_random_pred = svm.predict(X_random)
    accuracy_random = accuracy_score(y_random, y_random_pred)
    precision_random = precision_score(y_random, y_random_pred)
    recall_random = recall_score(y_random, y_random_pred)
    f1_random = f1_score(y_random, y_random_pred)

    # SINGLE-LINE PRINT for random-subset metrics
    print("[Feature Vectors][Random Values Testing] Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}"
          .format(accuracy_random, precision_random, recall_random, f1_random))

    # 7) Plot the data points (entire X, not just test)
    colors = np.where(y == 1, 'g', 'r')
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, edgecolor='k')

    # 8) Optional: show the decision boundary for 2D
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 300),
        np.linspace(ylim[0], ylim[1], 300)
    )
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles=['-'])

    # 9) Show standard main test metrics in plot title
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.title("Feature Vectors (MDS)\nAcc={:.2f}, Prec={:.2f}, Rec={:.2f}, F1={:.2f}"
              .format(accuracy_main, precision_main, recall_main, f1_main))

    plt.show()


# how to run: `python app.py %function_name%`
# example:
# python app.py process_data
# python app.py process_svm
# python app.py plot_results
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function_name', help='Name of function to run')
    args = parser.parse_args()

    if args.function_name == 'process_data':
        process_data()
    elif args.function_name == 'process_svm_default':
        process_svm_default()
    elif args.function_name == 'process_svm_feature_vectors':
        process_svm_feature_vectors()
    elif args.function_name == 'plot_results':
        plot_results()
    elif args.function_name == 'plot_clusters':
        plot_clusters()
    else:
        print(f'Error: Invalid function name "{args.function_name}"')
