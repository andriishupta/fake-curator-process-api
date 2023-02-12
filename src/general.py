from heapq import nlargest

from src.api.nlp import nlp


def filter_doc(doc):
    filtered_tokens = []

    # Iterate over the tokens in the processed text
    for token in doc:
        # Check if the token is a stop word, punctuation or contains some numbers/symbols
        if not token.is_stop:
            # If not, add the token to the filtered list
            filtered_tokens.append(token)

    body_clean_text = " ".join([token.text for token in filtered_tokens])

    return nlp(body_clean_text)


def summarize_doc(doc, selection_ratio=0.1):
    sentences = [sent.text for sent in doc.sents]
    word_frequencies = {}
    for sent in sentences:
        for word in sent:
            if word.text not in word_frequencies:
                word_frequencies[word.text] = 0
            word_frequencies[word.text] += 1
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)
    sentence_scores = {}
    for sent in sentences:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(" ")) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
    summarized_sentences = nlargest(int(len(sentences) * selection_ratio), sentence_scores, key=sentence_scores.get)

    return summarized_sentences


def top_bigram_frequency(doc, selection_ratio=0.1):
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


def top_lemma_frequency(doc, selection_ratio=0.1):
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
