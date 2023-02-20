from collections import Counter


def detect_biased_language(doc, subjectivity_ratio=0.1):
    # Initialize a list to store the biased words
    token_len = len(doc)
    subjective_words = []

    lemma_counts = {}
    for sent in doc.sents:
        for token in sent:
            lemma = token.lemma_
            lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1

    # Loop through each token in the document
    for token in doc:
        if token._.blob.subjectivity > subjectivity_ratio:
            subjective_words.append(token.text)

    subjective_words_ratio = len(subjective_words) / token_len

    top_subjective_words = dict(Counter(subjective_words).most_common(10))

    # Return the results
    return {
        "subjective_words_ratio": subjective_words_ratio,
        "top_subjective_words": top_subjective_words,
    }


DEHUMANIZING_LABELS = {"PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT"}


def detect_dehumanizing_language_ratio(doc):
    num_dehumanizing = 0
    num_total = 0

    for token in doc:
        if token.pos_ == "PROPN" and token.ent_type_ in DEHUMANIZING_LABELS:
            num_dehumanizing += 1
        num_total += 1

    if num_total == 0:
        return 0.0
    else:
        return num_dehumanizing / num_total


def lexical_diversity(lemma_counts, token):
    total_lemmas = sum(lemma_counts.values())

    if token.lemma_ in lemma_counts:
        return lemma_counts[token.lemma_] / total_lemmas
    else:
        return 0


def detect_paraphrased_ideas(doc):
    sents = [sent for sent in doc.sents]

    similarities = []
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            sim = sents[i].similarity(sents[j])
            similarities.append(sim)

    return sum(similarities) / len(similarities)
