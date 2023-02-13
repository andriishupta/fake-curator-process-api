from collections import Counter


def detect_biased_language(doc, subjectivity_ratio=0.5, lexical_diversity_ratio=0.5):
    # Initialize a list to store the biased words
    token_len = len(doc)
    subjective_words = []
    lexical_diversity_words = []

    lemma_counts = {}
    for sent in doc.sents:
        for token in sent:
            lemma = token.lemma_
            lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1

    # Loop through each token in the document
    for token in doc:
        if token._.blob.subjectivity > subjectivity_ratio:
            subjective_words.append(token.text)

        if lexical_diversity(lemma_counts, token) > lexical_diversity_ratio:
            lexical_diversity_words.append(token.text)

    subjective_words_ratio = len(subjective_words) / token_len
    lexical_diversity_words_ratio = len(lexical_diversity_words) / token_len

    top_subjective_words = dict(Counter(subjective_words).most_common(10))
    top_lexical_diversity_words = dict(Counter(lexical_diversity_words).most_common(10))

    # Return the results
    return {
        "subjective_words_ratio": subjective_words_ratio,
        "top_subjective_words": top_subjective_words,
        "lexical_diversity_words_ratio": lexical_diversity_words_ratio,
        "top_lexical_diversity_words": top_lexical_diversity_words,
    }


def detect_dehumanizing_language(doc):
    dehumanizing_entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE"]:
            for token in ent:
                if token._.blob.polarity < 0:
                    dehumanizing_entities.append(ent.text)
    return len(dehumanizing_entities) / len(doc) if len(doc) else 0


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
