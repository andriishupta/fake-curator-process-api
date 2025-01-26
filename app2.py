import csv
import json
import argparse
import random
import re
import os

import numpy as np
import pandas as pd

from sklearn.manifold import MDS
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.stem import WordNetLemmatizer
from nltk import download

from src.general import (
    clean_doc,
    summarize_text,
    top_bigram_frequency,
    top_lemma_frequency,
    extend_text_with_explanation,
)
from src.persuasion import (
    detect_biased_language,
    detect_paraphrased_ideas,
    detect_dehumanizing_language_ratio,
)
from src.sentiment import analyze_sentiment
from src.linguistic import (
    detect_unusual_inappropriate_language_ratio,
    detect_awkward_text_ratio,
)
from src.ner import get_ner_frequency

# Download necessary NLTK data
download('punkt')
download('wordnet')

CSV_LINES = 100


def process_data():
    """
    Runs the complete data processing pipeline:
    1. Processes CSV files to generate JSON outputs.
    2. Performs MDS dimensionality reduction.
    3. Combines TF-IDF and MDS feature vectors.
    """
    process_csv()
    process_mds()
    process_joined_method()


def process_csv(fake_csv_path='data/Fake.csv', true_csv_path='data/True.csv', output_dir='out', csv_lines=CSV_LINES):
    """
    Processes Fake and True CSV files, applies text processing, and saves the results as JSON files.

    Parameters:
    - fake_csv_path (str): Path to the Fake CSV file.
    - true_csv_path (str): Path to the True CSV file.
    - output_dir (str): Directory to save the output JSON files.
    - csv_lines (int): Number of lines to sample from each CSV.
    """
    lemmas = set()
    fake_data = []
    true_data = []

    def sample_and_process(file_path, label, data_list, sample_size):
        try:
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = list(csv.DictReader(csvfile))
                total_rows = len(reader)
                sampled = random.sample(reader, sample_size) if total_rows >= sample_size else reader
                for i, row in enumerate(sampled, 1):
                    title = row.get('title', '').strip()
                    text = row.get('text', '').strip()
                    if not title or not text:
                        continue
                    try:
                        print(f'Processing "{label}" #{i}; title: {title}')
                        processed = process_text(title, text)
                        lemmas.update(processed.get("general", {}).get("top_lemmas", []))
                        data_list.append(processed)
                    except Exception as e:
                        print(f'Error processing "{label}" #{i}: {e}')
                        continue
        except Exception as e:
            print(f'Error reading file {file_path}: {e}')

    sample_and_process(fake_csv_path, 'Fake', fake_data, csv_lines)
    sample_and_process(true_csv_path, 'True', true_data, csv_lines)

    random.shuffle(fake_data)
    random.shuffle(true_data)
    lemmas_sorted = sorted(lemmas)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'fake.json'), 'w', encoding='utf-8') as f:
        json.dump(fake_data, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, 'true.json'), 'w', encoding='utf-8') as f:
        json.dump(true_data, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, 'lemmas.json'), 'w', encoding='utf-8') as f:
        json.dump(list(lemmas_sorted), f, ensure_ascii=False, indent=4)

    print("Data processing complete. Output saved to 'out/' directory.")


def process_text(header_text, body_text):
    """
    Processes the text by performing various analyses and returns a structured dictionary.

    Parameters:
    - header_text (str): The title or header of the article.
    - body_text (str): The main content of the article.

    Returns:
    - dict: A dictionary containing processed data.
    """
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
    persuasion_data = {
        "paraphrased_ratio": detect_paraphrased_ideas(body_doc),
        "dehumanizing_language_ratio": detect_dehumanizing_language_ratio(body_doc),
        **detect_biased_language(body_doc)
    }

    print("# -----------Narrative-------------- #")
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

    print("# -----------Sentiment Analysis-------------- #")
    sentiment_analysis_data = analyze_sentiment(body_doc)

    processed_data = {
        "text": body_text,  # Added to prevent KeyError
        "general": general_data,
        "persuasion": persuasion_data,
        "narrative": narrative_data,
        "linguistic": linguistic_data,
        "sentiment_analysis": sentiment_analysis_data,
    }

    print("# -----------Post Process-------------- #")
    processed_data["extended_text_with_explanation"] = extend_text_with_explanation(body_text, processed_data)

    return processed_data


def process_joined_method(fake_json='out/fake.json', true_json='out/true.json', mds_json='out/mds_results.json', output_dir='out', max_features=5000):
    """
    Combines TF-IDF and MDS feature vectors for SVM classification.

    Parameters:
    - fake_json (str): Path to fake.json file.
    - true_json (str): Path to true.json file.
    - mds_json (str): Path to mds_results.json file.
    - output_dir (str): Directory to save output files (if needed).
    - max_features (int): Maximum number of TF-IDF features.

    Returns:
    - X_combined (numpy array): Combined TF-IDF and MDS feature vectors.
    - y (numpy array): Corresponding labels (0 for fake, 1 for true).
    """
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Text preprocessing function
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmas)

    # Load and preprocess fake data
    try:
        with open(fake_json, 'r', encoding='utf-8') as f:
            fake_data = json.load(f)
    except Exception as e:
        print(f'Error loading {fake_json}: {e}')
        fake_data = []
    fake_texts = [preprocess(item['text']) for item in fake_data]
    fake_labels = [0] * len(fake_texts)

    # Load and preprocess true data
    try:
        with open(true_json, 'r', encoding='utf-8') as f:
            true_data = json.load(f)
    except Exception as e:
        print(f'Error loading {true_json}: {e}')
        true_data = []
    true_texts = [preprocess(item['text']) for item in true_data]
    true_labels = [1] * len(true_texts)

    # Combine data
    texts = fake_texts + true_texts
    labels = fake_labels + true_labels
    y = np.array(labels)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(texts).toarray()  # Convert to dense for concatenation

    # Load MDS features
    try:
        with open(mds_json, 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
        X_mds = np.array(mds_results["mds"])
        if X_mds.shape[0] != len(texts):
            print("Warning: Number of MDS features does not match number of texts.")
            print(f"Number of MDS samples: {X_mds.shape[0]}, Number of texts: {len(texts)}")
    except Exception as e:
        print(f'Error loading {mds_json}: {e}')
        X_mds = np.zeros((len(texts), 0))  # Empty array if error

    # Validation: Check if sample sizes match
    if X_tfidf.shape[0] != X_mds.shape[0]:
        print(f"Error: Mismatch in sample sizes between TF-IDF ({X_tfidf.shape[0]}) and MDS ({X_mds.shape[0]}).")
        print("Please ensure that 'process_csv' and 'process_mds' are run with the same number of samples.")
        return None, None

    # Combine TF-IDF and MDS features
    if X_mds.size > 0:
        X_combined = np.hstack((X_tfidf, X_mds))
    else:
        X_combined = X_tfidf

    # (Optional) Save the combined feature vectors
    # os.makedirs(output_dir, exist_ok=True)
    # np.save(os.path.join(output_dir, 'combined_features.npy'), X_combined)
    # np.save(os.path.join(output_dir, 'labels.npy'), y)

    print(f"Combined TF-IDF and MDS feature vectors created with shape: {X_combined.shape}")
    return X_combined, y


def process_mds():
    """
    Performs Multi-Dimensional Scaling (MDS) on processed data and saves the results.
    """
    # Load processed data from fake.json and true.json
    try:
        with open('out/fake.json', 'r', encoding='utf-8') as f:
            fake_data = json.load(f)
    except Exception as e:
        print(f'Error loading fake.json: {e}')
        fake_data = []
    try:
        with open('out/true.json', 'r', encoding='utf-8') as f:
            true_data = json.load(f)
    except Exception as e:
        print(f'Error loading true.json: {e}')
        true_data = []
    try:
        with open('out/lemmas.json', 'r', encoding='utf-8') as f:
            lemmas = json.load(f)
    except Exception as e:
        print(f'Error loading lemmas.json: {e}')
        lemmas = []

    fake_values = []
    for item in fake_data:
        try:
            item_values = [
                item['persuasion']['paraphrased_ratio'],
                # item['persuasion']['dehumanizing_language_ratio'],
                # item['persuasion']['subjective_words_ratio'],

                # item['narrative']['header_summary_similarity_ratio'],

                item['linguistic']['unusual_inappropriate_language_ratio'],
                # item['linguistic']['awkward_text_ratio'],

                item['sentiment_analysis']['avg_sentiment'],
                # item['sentiment_analysis']['positive_ratio'],
                # item['sentiment_analysis']['neutral_ratio'],
                # item['sentiment_analysis']['negative_ratio'],

                # *lemmas_params(lemmas, item['general']['lemma_frequency'])
            ]
            fake_values.append(item_values)
        except KeyError as e:
            print(f'Missing key in fake_data: {e}')
            continue

    true_values = []
    for item in true_data:
        try:
            item_values = [
                item['persuasion']['paraphrased_ratio'],
                # item['persuasion']['dehumanizing_language_ratio'],
                # item['persuasion']['subjective_words_ratio'],

                # item['narrative']['header_summary_similarity_ratio'],

                item['linguistic']['unusual_inappropriate_language_ratio'],
                # item['linguistic']['awkward_text_ratio'],

                item['sentiment_analysis']['avg_sentiment'],
                # item['sentiment_analysis']['positive_ratio'],
                # item['sentiment_analysis']['neutral_ratio'],
                # item['sentiment_analysis']['negative_ratio'],

                # *lemmas_params(lemmas, item['general']['lemma_frequency'])
            ]
            true_values.append(item_values)
        except KeyError as e:
            print(f'Missing key in true_data: {e}')
            continue

    combined_values = true_values + fake_values
    if not combined_values:
        print("No data to process for MDS.")
        return

    mds = MDS(random_state=42).fit_transform(np.array(combined_values))

    # Save output to mds_results.json
    os.makedirs('out', exist_ok=True)
    with open('out/mds_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            "mds": mds.tolist(),
            "true": len(true_values),
            "fake": len(fake_values),
        }, f, ensure_ascii=False, indent=4)

    print("MDS processing complete. Results saved to 'out/mds_results.json'.")


def lemmas_params(all_lemmas, item_lemmas_dict):
    """
    Generates a list of lemma frequencies based on the global lemmas list.

    Parameters:
    - all_lemmas (list): Sorted list of all lemmas.
    - item_lemmas_dict (dict): Dictionary of lemma frequencies for an item.

    Returns:
    - list: Frequencies corresponding to all_lemmas.
    """
    params = []
    for lemma in all_lemmas:
        params.append(item_lemmas_dict.get(lemma, 0))
    return params


def plot_results():
    """
    Plots the MDS results, coloring points based on their labels.
    """
    try:
        with open('out/mds_results.json', 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
    except Exception as e:
        print(f'Error loading mds_results.json: {e}')
        return

    X = np.array(mds_results["mds"])
    y = np.concatenate((np.ones(mds_results["true"]), np.zeros(mds_results["fake"])))

    colors = np.where(y == 1, 'g', 'r')

    # Plot true MDS values in green and fake MDS values in red
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, edgecolor='k')

    # Add axis labels and title
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.title("Fake/Real News Visualization")

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='g', edgecolor='k', label='True'),
                       Patch(facecolor='r', edgecolor='k', label='Fake')]
    plt.legend(handles=legend_elements)

    # Show the plot
    plt.show()


def plot_clusters():
    """
    Performs K-Means clustering on MDS results and plots the clusters with centroids.
    """
    try:
        with open('out/mds_results.json', 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
    except Exception as e:
        print(f'Error loading mds_results.json: {e}')
        return

    X = np.array(mds_results["mds"])
    y = np.concatenate((np.ones(mds_results["true"]), np.zeros(mds_results["fake"])))

    # Perform k-means clustering
    k = 2  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_

    # Generate scatter plot of MDS data
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set1', alpha=0.6, edgecolor='k')

    # Add plot title and legend
    plt.title('MDS Plot with K-Means Clustering')
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")

    # Plot centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=3, color='black', label='Centroids')

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='C0', edgecolor='k', label='Cluster 0'),
                       Patch(facecolor='C1', edgecolor='k', label='Cluster 1'),
                       Patch(facecolor='black', edgecolor='k', label='Centroids')]
    plt.legend(handles=legend_elements)

    # Show the plot
    plt.show()


def process_svm_default():
    """
    Trains and evaluates an SVM model using default TF-IDF features.
    """
    try:
        true = pd.read_csv("data/True.csv")
        true['label'] = 1
    except Exception as e:
        print(f'Error loading True.csv: {e}')
        true = pd.DataFrame(columns=['text', 'label'])

    try:
        fake = pd.read_csv("data/Fake.csv")
        fake['label'] = 0
    except Exception as e:
        print(f'Error loading Fake.csv: {e}')
        fake = pd.DataFrame(columns=['text', 'label'])

    df = pd.concat([true.head(CSV_LINES), fake.head(CSV_LINES)])
    X = df['text'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Extract features using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the SVM model
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_tfidf, y_train)

    # Predict probabilities and classes on the test set
    y_pred = svm.predict(X_test_tfidf)
    y_prob = svm.predict_proba(X_test_tfidf)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n[Default] Evaluation Metrics on Test Set:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    # Generate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Default TF-IDF")
    plt.show()

    # Generate and plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Default TF-IDF')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Generate and plot the Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='purple', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Default TF-IDF')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True'], zero_division=0))


def process_svm_feature_vectors():
    """
    Trains and evaluates an SVM model using MDS feature vectors.
    """
    # 1) Load MDS results
    try:
        with open('out/mds_results.json', 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
    except Exception as e:
        print(f'Error loading mds_results.json: {e}')
        return

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
        random_state=42,
        stratify=y
    )

    # 4) Train an SVM model
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    # 5) Predict probabilities and classes on the test set
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]

    # 6) Calculate evaluation metrics
    accuracy_main = accuracy_score(y_test, y_pred)
    precision_main = precision_score(y_test, y_pred, zero_division=0)
    recall_main = recall_score(y_test, y_pred, zero_division=0)
    f1_main = f1_score(y_test, y_pred, zero_division=0)

    print("\n[Feature Vectors] Evaluation Metrics on Test Set:")
    print(f"Accuracy : {accuracy_main:.4f}")
    print(f"Precision: {precision_main:.4f}")
    print(f"Recall   : {recall_main:.4f}")
    print(f"F1 Score : {f1_main:.4f}\n")

    # 7) Generate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - MDS Features")
    plt.show()

    # 8) Generate and plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - MDS Features')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # 9) Generate and plot the Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='purple', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - MDS Features')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # 10) Display classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True'], zero_division=0))


def process_svm_joined_method():
    """
    Trains and evaluates an SVM model using combined TF-IDF and MDS feature vectors.
    Includes comprehensive plotting for performance visualization.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

    # Load combined features and labels
    X, y = process_joined_method()

    # If process_joined_method returned None, exit
    if X is None or y is None:
        print("Exiting due to previous errors.")
        return

    # Check the shape of the data
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y.astype(int))}")

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    # Initialize and train the SVM model with probability estimates
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    # Predict probabilities and classes on the test set
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n[Joined Method] Evaluation Metrics on Test Set:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    # Generate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Joined Features")
    plt.show()

    # Generate and plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Joined Features')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Generate and plot the Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='purple', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Joined Features')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True'], zero_division=0))

    # Plotting Decision Boundary (if possible)
    # Note: This is feasible only if the feature space is 2D (e.g., using only MDS dimensions)
    if X.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette=['red', 'green'], alpha=0.6, edgecolor='k')

        # Create a mesh to plot the decision boundary
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.contour(xx, yy, Z, levels=[0.5], cmap="Greys", vmin=0, vmax=.6)
        plt.title("SVM Decision Boundary with Joined Features")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend(title='True Label', labels=['Fake', 'True'])
        plt.show()
    else:
        print("Decision boundary plot is skipped because the feature space is not 2D.")

    # Investigate Perfect Scores
    if accuracy == 1.0 and precision == 1.0 and recall == 1.0 and f1 == 1.0:
        print("Warning: The model achieved perfect scores. This may indicate overfitting or data leakage.")
        print("Consider the following checks:")
        print("1. Ensure that there is no overlap between training and testing data.")
        print("2. Verify that the feature combination does not include any data leakage.")
        print("3. Evaluate the model using cross-validation for more reliable metrics.")
    else:
        print("Model evaluation completed successfully.")


def process_svm_joined_method_with_cv():
    """
    Trains and evaluates an SVM model using combined TF-IDF and MDS feature vectors with cross-validation.
    Includes comprehensive plotting for performance visualization.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

    # Load combined features and labels
    X, y = process_joined_method()

    # If process_joined_method returned None, exit
    if X is None or y is None:
        print("Exiting due to previous errors.")
        return

    # Check the shape of the data
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y.astype(int))}")

    # Initialize the SVM model with probability estimates
    svm = SVC(kernel='linear', probability=True, random_state=42)

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(svm, X, y, cv=cv, scoring='f1')
    print(f"Cross-Validation F1 Scores: {cv_scores}")
    print(f"Average Cross-Validation F1 Score: {cv_scores.mean():.4f}\n")

    # Fit the model on the entire dataset
    svm.fit(X, y)

    # Predict probabilities and classes on the entire dataset
    y_pred = svm.predict(X)
    y_prob = svm.predict_proba(X)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print("\n[Joined Method with CV] Evaluation Metrics on Entire Dataset:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    # Generate and display the confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Joined Features (Entire Dataset)")
    plt.show()

    # Generate and plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Joined Features (Entire Dataset)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Generate and plot the Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='purple', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Joined Features (Entire Dataset)')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # Display classification report
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=['Fake', 'True'], zero_division=0))

    # Plotting Decision Boundary (if possible)
    # Note: This is feasible only if the feature space is 2D (e.g., using only MDS dimensions)
    if X.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=['red', 'green'], alpha=0.6, edgecolor='k')

        # Create a mesh to plot the decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.contour(xx, yy, Z, levels=[0.5], cmap="Greys", vmin=0, vmax=.6)
        plt.title("SVM Decision Boundary with Joined Features (Entire Dataset)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend(title='True Label', labels=['Fake', 'True'])
        plt.show()
    else:
        print("Decision boundary plot is skipped because the feature space is not 2D.")

    # Investigate Perfect Scores
    if accuracy == 1.0 and precision == 1.0 and recall == 1.0 and f1 == 1.0:
        print("Warning: The model achieved perfect scores. This may indicate overfitting or data leakage.")
        print("Consider the following checks:")
        print("1. Ensure that there is no overlap between training and testing data.")
        print("2. Verify that the feature combination does not include any data leakage.")
        print("3. Evaluate the model using cross-validation for more reliable metrics.")
    else:
        print("Model evaluation completed successfully.")


def process_joined_method(fake_json='out/fake.json', true_json='out/true.json', mds_json='out/mds_results.json', output_dir='out', max_features=5000):
    """
    Combines TF-IDF and MDS feature vectors for SVM classification.

    Parameters:
    - fake_json (str): Path to fake.json file.
    - true_json (str): Path to true.json file.
    - mds_json (str): Path to mds_results.json file.
    - output_dir (str): Directory to save output files (if needed).
    - max_features (int): Maximum number of TF-IDF features.

    Returns:
    - X_combined (numpy array): Combined TF-IDF and MDS feature vectors.
    - y (numpy array): Corresponding labels (0 for fake, 1 for true).
    """
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Text preprocessing function
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmas)

    # Load and preprocess fake data
    try:
        with open(fake_json, 'r', encoding='utf-8') as f:
            fake_data = json.load(f)
    except Exception as e:
        print(f'Error loading {fake_json}: {e}')
        fake_data = []
    fake_texts = [preprocess(item['text']) for item in fake_data]
    fake_labels = [0] * len(fake_texts)

    # Load and preprocess true data
    try:
        with open(true_json, 'r', encoding='utf-8') as f:
            true_data = json.load(f)
    except Exception as e:
        print(f'Error loading {true_json}: {e}')
        true_data = []
    true_texts = [preprocess(item['text']) for item in true_data]
    true_labels = [1] * len(true_texts)

    # Combine data
    texts = fake_texts + true_texts
    labels = fake_labels + true_labels
    y = np.array(labels)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(texts).toarray()  # Convert to dense for concatenation

    # Load MDS features
    try:
        with open(mds_json, 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
        X_mds = np.array(mds_results["mds"])
        if X_mds.shape[0] != len(texts):
            print("Warning: Number of MDS features does not match number of texts.")
            print(f"Number of MDS samples: {X_mds.shape[0]}, Number of texts: {len(texts)}")
    except Exception as e:
        print(f'Error loading {mds_json}: {e}')
        X_mds = np.zeros((len(texts), 0))  # Empty array if error

    # Validation: Check if sample sizes match
    if X_tfidf.shape[0] != X_mds.shape[0]:
        print(f"Error: Mismatch in sample sizes between TF-IDF ({X_tfidf.shape[0]}) and MDS ({X_mds.shape[0]}).")
        print("Please ensure that 'process_csv' and 'process_mds' are run with the same number of samples.")
        return None, None

    # Combine TF-IDF and MDS features
    if X_mds.size > 0:
        X_combined = np.hstack((X_tfidf, X_mds))
    else:
        X_combined = X_tfidf

    # (Optional) Save the combined feature vectors
    # os.makedirs(output_dir, exist_ok=True)
    # np.save(os.path.join(output_dir, 'combined_features.npy'), X_combined)
    # np.save(os.path.join(output_dir, 'labels.npy'), y)

    print(f"Combined TF-IDF and MDS feature vectors created with shape: {X_combined.shape}")
    return X_combined, y


def plot_results():
    """
    Plots the MDS results, coloring points based on their labels.
    """
    try:
        with open('out/mds_results.json', 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
    except Exception as e:
        print(f'Error loading mds_results.json: {e}')
        return

    X = np.array(mds_results["mds"])
    y = np.concatenate((np.ones(mds_results["true"]), np.zeros(mds_results["fake"])))

    colors = np.where(y == 1, 'g', 'r')

    # Plot true MDS values in green and fake MDS values in red
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, edgecolor='k')

    # Add axis labels and title
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.title("Fake/Real News Visualization")

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='g', edgecolor='k', label='True'),
                       Patch(facecolor='r', edgecolor='k', label='Fake')]
    plt.legend(handles=legend_elements)

    # Show the plot
    plt.show()


def plot_clusters():
    """
    Performs K-Means clustering on MDS results and plots the clusters with centroids.
    """
    try:
        with open('out/mds_results.json', 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
    except Exception as e:
        print(f'Error loading mds_results.json: {e}')
        return

    X = np.array(mds_results["mds"])
    y = np.concatenate((np.ones(mds_results["true"]), np.zeros(mds_results["fake"])))

    # Perform k-means clustering
    k = 2  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_

    # Generate scatter plot of MDS data
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set1', alpha=0.6, edgecolor='k')

    # Add plot title and legend
    plt.title('MDS Plot with K-Means Clustering')
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")

    # Plot centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=3, color='black', label='Centroids')

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='C0', edgecolor='k', label='Cluster 0'),
                       Patch(facecolor='C1', edgecolor='k', label='Cluster 1'),
                       Patch(facecolor='black', edgecolor='k', label='Centroids')]
    plt.legend(handles=legend_elements)

    # Show the plot
    plt.show()


def process_svm_default():
    """
    Trains and evaluates an SVM model using default TF-IDF features.
    """
    try:
        true = pd.read_csv("data/True.csv")
        true['label'] = 1
    except Exception as e:
        print(f'Error loading True.csv: {e}')
        true = pd.DataFrame(columns=['text', 'label'])

    try:
        fake = pd.read_csv("data/Fake.csv")
        fake['label'] = 0
    except Exception as e:
        print(f'Error loading Fake.csv: {e}')
        fake = pd.DataFrame(columns=['text', 'label'])

    df = pd.concat([true.head(CSV_LINES), fake.head(CSV_LINES)])
    X = df['text'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Extract features using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the SVM model
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_tfidf, y_train)

    # Predict probabilities and classes on the test set
    y_pred = svm.predict(X_test_tfidf)
    y_prob = svm.predict_proba(X_test_tfidf)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n[Default] Evaluation Metrics on Test Set:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    # Generate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Default TF-IDF")
    plt.show()

    # Generate and plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Default TF-IDF')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Generate and plot the Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='purple', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Default TF-IDF')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True'], zero_division=0))


def process_svm_feature_vectors():
    """
    Trains and evaluates an SVM model using MDS feature vectors.
    """
    # 1) Load MDS results
    try:
        with open('out/mds_results.json', 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
    except Exception as e:
        print(f'Error loading mds_results.json: {e}')
        return

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
        random_state=42,
        stratify=y
    )

    # 4) Train an SVM model
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    # 5) Predict probabilities and classes on the test set
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]

    # 6) Calculate evaluation metrics
    accuracy_main = accuracy_score(y_test, y_pred)
    precision_main = precision_score(y_test, y_pred, zero_division=0)
    recall_main = recall_score(y_test, y_pred, zero_division=0)
    f1_main = f1_score(y_test, y_pred, zero_division=0)

    print("\n[Feature Vectors] Evaluation Metrics on Test Set:")
    print(f"Accuracy : {accuracy_main:.4f}")
    print(f"Precision: {precision_main:.4f}")
    print(f"Recall   : {recall_main:.4f}")
    print(f"F1 Score : {f1_main:.4f}\n")

    # 7) Generate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - MDS Features")
    plt.show()

    # 8) Generate and plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - MDS Features')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # 9) Generate and plot the Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='purple', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - MDS Features')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # 10) Display classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True'], zero_division=0))


def process_svm_joined_method():
    """
    Trains and evaluates an SVM model using combined TF-IDF and MDS feature vectors.
    Includes comprehensive plotting for performance visualization.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

    # Load combined features and labels
    X, y = process_joined_method()

    # If process_joined_method returned None, exit
    if X is None or y is None:
        print("Exiting due to previous errors.")
        return

    # Check the shape of the data
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y.astype(int))}")

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    # Initialize and train the SVM model with probability estimates
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    # Predict probabilities and classes on the test set
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n[Joined Method] Evaluation Metrics on Test Set:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    # Generate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Joined Features")
    plt.show()

    # Generate and plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Joined Features')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Generate and plot the Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='purple', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Joined Features')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True'], zero_division=0))

    # Plotting Decision Boundary (if possible)
    # Note: This is feasible only if the feature space is 2D (e.g., using only MDS dimensions)
    if X.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette=['red', 'green'], alpha=0.6, edgecolor='k')

        # Create a mesh to plot the decision boundary
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.contour(xx, yy, Z, levels=[0.5], cmap="Greys", vmin=0, vmax=.6)
        plt.title("SVM Decision Boundary with Joined Features")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend(title='True Label', labels=['Fake', 'True'])
        plt.show()
    else:
        print("Decision boundary plot is skipped because the feature space is not 2D.")

    # Investigate Perfect Scores
    if accuracy == 1.0 and precision == 1.0 and recall == 1.0 and f1 == 1.0:
        print("Warning: The model achieved perfect scores. This may indicate overfitting or data leakage.")
        print("Consider the following checks:")
        print("1. Ensure that there is no overlap between training and testing data.")
        print("2. Verify that the feature combination does not include any data leakage.")
        print("3. Evaluate the model using cross-validation for more reliable metrics.")
    else:
        print("Model evaluation completed successfully.")


def process_svm_joined_method_with_cv():
    """
    Trains and evaluates an SVM model using combined TF-IDF and MDS feature vectors with cross-validation.
    Includes comprehensive plotting for performance visualization.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

    # Load combined features and labels
    X, y = process_joined_method()

    # If process_joined_method returned None, exit
    if X is None or y is None:
        print("Exiting due to previous errors.")
        return

    # Check the shape of the data
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y.astype(int))}")

    # Initialize the SVM model with probability estimates
    svm = SVC(kernel='linear', probability=True, random_state=42)

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(svm, X, y, cv=cv, scoring='f1')
    print(f"Cross-Validation F1 Scores: {cv_scores}")
    print(f"Average Cross-Validation F1 Score: {cv_scores.mean():.4f}\n")

    # Fit the model on the entire dataset
    svm.fit(X, y)

    # Predict probabilities and classes on the entire dataset
    y_pred = svm.predict(X)
    y_prob = svm.predict_proba(X)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print("\n[Joined Method with CV] Evaluation Metrics on Entire Dataset:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    # Generate and display the confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Joined Features (Entire Dataset)")
    plt.show()

    # Generate and plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Joined Features (Entire Dataset)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Generate and plot the Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='purple', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Joined Features (Entire Dataset)')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # Display classification report
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=['Fake', 'True'], zero_division=0))

    # Plotting Decision Boundary (if possible)
    # Note: This is feasible only if the feature space is 2D (e.g., using only MDS dimensions)
    if X.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=['red', 'green'], alpha=0.6, edgecolor='k')

        # Create a mesh to plot the decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.contour(xx, yy, Z, levels=[0.5], cmap="Greys", vmin=0, vmax=.6)
        plt.title("SVM Decision Boundary with Joined Features (Entire Dataset)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend(title='True Label', labels=['Fake', 'True'])
        plt.show()
    else:
        print("Decision boundary plot is skipped because the feature space is not 2D.")

    # Investigate Perfect Scores
    if accuracy == 1.0 and precision == 1.0 and recall == 1.0 and f1 == 1.0:
        print("Warning: The model achieved perfect scores. This may indicate overfitting or data leakage.")
        print("Consider the following checks:")
        print("1. Ensure that there is no overlap between training and testing data.")
        print("2. Verify that the feature combination does not include any data leakage.")
        print("3. Evaluate the model using cross-validation for more reliable metrics.")
    else:
        print("Model evaluation completed successfully.")


def process_joined_method(fake_json='out/fake.json', true_json='out/true.json', mds_json='out/mds_results.json', output_dir='out', max_features=5000):
    """
    Combines TF-IDF and MDS feature vectors for SVM classification.

    Parameters:
    - fake_json (str): Path to fake.json file.
    - true_json (str): Path to true.json file.
    - mds_json (str): Path to mds_results.json file.
    - output_dir (str): Directory to save output files (if needed).
    - max_features (int): Maximum number of TF-IDF features.

    Returns:
    - X_combined (numpy array): Combined TF-IDF and MDS feature vectors.
    - y (numpy array): Corresponding labels (0 for fake, 1 for true).
    """
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Text preprocessing function
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmas)

    # Load and preprocess fake data
    try:
        with open(fake_json, 'r', encoding='utf-8') as f:
            fake_data = json.load(f)
    except Exception as e:
        print(f'Error loading {fake_json}: {e}')
        fake_data = []
    fake_texts = [preprocess(item['text']) for item in fake_data]
    fake_labels = [0] * len(fake_texts)

    # Load and preprocess true data
    try:
        with open(true_json, 'r', encoding='utf-8') as f:
            true_data = json.load(f)
    except Exception as e:
        print(f'Error loading {true_json}: {e}')
        true_data = []
    true_texts = [preprocess(item['text']) for item in true_data]
    true_labels = [1] * len(true_texts)

    # Combine data
    texts = fake_texts + true_texts
    labels = fake_labels + true_labels
    y = np.array(labels)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(texts).toarray()  # Convert to dense for concatenation

    # Load MDS features
    try:
        with open(mds_json, 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
        X_mds = np.array(mds_results["mds"])
        if X_mds.shape[0] != len(texts):
            print("Warning: Number of MDS features does not match number of texts.")
            print(f"Number of MDS samples: {X_mds.shape[0]}, Number of texts: {len(texts)}")
    except Exception as e:
        print(f'Error loading {mds_json}: {e}')
        X_mds = np.zeros((len(texts), 0))  # Empty array if error

    # Validation: Check if sample sizes match
    if X_tfidf.shape[0] != X_mds.shape[0]:
        print(f"Error: Mismatch in sample sizes between TF-IDF ({X_tfidf.shape[0]}) and MDS ({X_mds.shape[0]}).")
        print("Please ensure that 'process_csv' and 'process_mds' are run with the same number of samples.")
        return None, None

    # Combine TF-IDF and MDS features
    if X_mds.size > 0:
        X_combined = np.hstack((X_tfidf, X_mds))
    else:
        X_combined = X_tfidf

    # (Optional) Save the combined feature vectors
    # os.makedirs(output_dir, exist_ok=True)
    # np.save(os.path.join(output_dir, 'combined_features.npy'), X_combined)
    # np.save(os.path.join(output_dir, 'labels.npy'), y)

    print(f"Combined TF-IDF and MDS feature vectors created with shape: {X_combined.shape}")
    return X_combined, y


def plot_results():
    """
    Plots the MDS results, coloring points based on their labels.
    """
    try:
        with open('out/mds_results.json', 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
    except Exception as e:
        print(f'Error loading mds_results.json: {e}')
        return

    X = np.array(mds_results["mds"])
    y = np.concatenate((np.ones(mds_results["true"]), np.zeros(mds_results["fake"])))

    colors = np.where(y == 1, 'g', 'r')

    # Plot true MDS values in green and fake MDS values in red
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, edgecolor='k')

    # Add axis labels and title
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.title("Fake/Real News Visualization")

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='g', edgecolor='k', label='True'),
                       Patch(facecolor='r', edgecolor='k', label='Fake')]
    plt.legend(handles=legend_elements)

    # Show the plot
    plt.show()


def plot_clusters():
    """
    Performs K-Means clustering on MDS results and plots the clusters with centroids.
    """
    try:
        with open('out/mds_results.json', 'r', encoding='utf-8') as f:
            mds_results = json.load(f)
    except Exception as e:
        print(f'Error loading mds_results.json: {e}')
        return

    X = np.array(mds_results["mds"])
    y = np.concatenate((np.ones(mds_results["true"]), np.zeros(mds_results["fake"])))

    # Perform k-means clustering
    k = 2  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_

    # Generate scatter plot of MDS data
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set1', alpha=0.6, edgecolor='k')

    # Add plot title and legend
    plt.title('MDS Plot with K-Means Clustering')
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")

    # Plot centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=3, color='black', label='Centroids')

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='C0', edgecolor='k', label='Cluster 0'),
                       Patch(facecolor='C1', edgecolor='k', label='Cluster 1'),
                       Patch(facecolor='black', edgecolor='k', label='Centroids')]
    plt.legend(handles=legend_elements)

    # Show the plot
    plt.show()


# how to run: `python app.py %function_name%`
# example:
# python app.py process_data
# python app.py process_svm_default
# python app.py process_svm_feature_vectors
# python app.py process_svm_joined_method
# python app.py process_svm_joined_method_with_cv
# python app.py plot_results
# python app.py plot_clusters
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function_name', help='Name of function to run')
    args = parser.parse_args()

    if args.function_name == 'process_data':
        process_data()
    elif args.function_name == 'process_csv':
        process_csv()
    elif args.function_name == 'process_joined_method':
        process_joined_method()
    elif args.function_name == 'process_mds':
        process_mds()
    elif args.function_name == 'process_svm_default':
        process_svm_default()
    elif args.function_name == 'process_svm_feature_vectors':
        process_svm_feature_vectors()
    elif args.function_name == 'process_svm_joined_method':
        process_svm_joined_method()
    elif args.function_name == 'process_svm_joined_method_with_cv':
        process_svm_joined_method_with_cv()
    elif args.function_name == 'plot_results':
        plot_results()
    elif args.function_name == 'plot_clusters':
        plot_clusters()
    else:
        print(f'Error: Invalid function name "{args.function_name}"')
