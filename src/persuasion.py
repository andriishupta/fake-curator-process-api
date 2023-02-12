from collections import Counter


def detect_biased_language(doc, subjectivity_ratio=0.5, lexical_diversity_ratio=0.7):
    # Initialize a list to store the biased words
    token_len = len(doc)
    subjective_biased_words = []
    lexical_diversity_biased_words = []

    # Loop through each token in the document
    for token in doc:
        if token._.subjectivity > subjectivity_ratio:
            subjective_biased_words.append(token.text)

        if token._.lex_div > lexical_diversity_ratio:
            lexical_diversity_biased_words.append(token.text)

    subjective_biased_words_ratio = len(subjective_biased_words) / token_len
    lexical_diversity_biased_words_ratio = len(lexical_diversity_biased_words) / token_len

    top_subjective_biased_words = Counter(subjective_biased_words).most_common(10)
    top_lexical_diversity_biased_words = Counter(lexical_diversity_biased_words).most_common(10)

    # Return the results
    return {
        "subjective_biased_words_ratio": subjective_biased_words_ratio,
        "lexical_diversity_biased_words_ratio": lexical_diversity_biased_words_ratio,
        "top_subjective_biased_words": top_subjective_biased_words,
        "top_lexical_diversity_biased_words": top_lexical_diversity_biased_words,
    }


def detect_bigram_frequency(doc, selection_ratio=0.1):
    bigram_freq = {}

    # Iterate over the tokens in the processed text
    for i in range(len(doc) - 1):
        # Get the current and next token
        current_token = doc[i]
        next_token = doc[i + 1]

        # Get the text for each token
        current_text = current_token.text.lower()
        next_text = next_token.text.lower()

        # Combine the text of the two tokens to form a bigram
        bigram = current_text + " " + next_text

        # Check if the bigram is already in the dictionary
        if bigram in bigram_freq:
            # If it is, increment the frequency
            bigram_freq[bigram] += 1
        else:
            # If not, add the bigram to the dictionary with frequency 1
            bigram_freq[bigram] = 1

    sorted_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
    top = int(len(sorted_bigrams) * selection_ratio)

    return dict(sorted_bigrams[:top])


def detect_lemma_frequency(doc, selection_ratio=0.1):
    lemma_freq = {}

    # Iterate over the tokens in the processed text
    for token in doc:
        # Get the lemma for each token
        lemma = token.lemma_.lower()

        # Check if the lemma is already in the dictionary
        if lemma in lemma_freq:
            # If it is, increment the frequency
            lemma_freq[lemma] += 1
        else:
            # If not, add the lemma to the dictionary with frequency 1
            lemma_freq[lemma] = 1

    sorted_lemmas = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
    top = int(len(sorted_lemmas) * selection_ratio)

    return dict(sorted_lemmas[:top])
