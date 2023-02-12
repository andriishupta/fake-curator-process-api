def detect_bigram_frequency(doc):
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

    return bigram_freq


def detect_lemma_frequency(doc):
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

    return lemma_freq
