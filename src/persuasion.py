from collections import Counter


def detect_biased_language(doc, subjectivity_ratio=0.5, lexical_diversity_ratio=0.5, polarity_ratio=0.5):
    # Initialize a list to store the biased words
    token_len = len(doc)
    subjective_biased_words = []
    polarity_biased_words = []
    lexical_diversity_biased_words = []

    lemma_counts = {}
    for sent in doc.sents:
        for token in sent:
            lemma = token.lemma_
            lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1

    # Loop through each token in the document
    for token in doc:
        if token._.blob.subjectivity > subjectivity_ratio:
            subjective_biased_words.append(token.text)

        if token._.blob.polarity > polarity_ratio:
            polarity_biased_words.append(token.text)

        if lexical_diversity(lemma_counts, token) > lexical_diversity_ratio:
            lexical_diversity_biased_words.append(token.text)

    subjective_biased_words_ratio = len(subjective_biased_words) / token_len
    polarity_biased_words_ratio = len(polarity_biased_words) / token_len
    lexical_diversity_biased_words_ratio = len(lexical_diversity_biased_words) / token_len

    top_subjective_biased_words = Counter(subjective_biased_words).most_common(10)
    top_polarity_words = Counter(polarity_biased_words).most_common(10)
    top_lexical_diversity_biased_words = Counter(lexical_diversity_biased_words).most_common(10)

    # Return the results
    return {
        "subjective_biased_words_ratio": subjective_biased_words_ratio,
        "polarity_biased_words_ratio": polarity_biased_words_ratio,
        "lexical_diversity_biased_words_ratio": lexical_diversity_biased_words_ratio,
        "top_subjective_biased_words": top_subjective_biased_words,
        "top_polarity_words": top_polarity_words,
        "top_lexical_diversity_biased_words": top_lexical_diversity_biased_words,
    }


def lexical_diversity(lemma_counts, token):
    total_lemmas = sum(lemma_counts.values())

    if token.lemma_ in lemma_counts:
        return lemma_counts[token.lemma_] / total_lemmas
    else:
        return 0


def detect_paraphrased_ideas(doc, similarity_threshold=0.5):
    total_sents = len(list(doc.sents))
    similar_sents = 0
    for sent1 in doc.sents:
        for sent2 in doc.sents:
            if sent1.similarity(sent2) > similarity_threshold:
                similar_sents += 1
                break
    return similar_sents / total_sents
