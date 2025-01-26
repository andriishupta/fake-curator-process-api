from src.api.nlp import nlp


def clean_doc(text):
    doc = nlp(text)

    processed_sents = []
    for sent in doc.sents:
        processed_tokens = []
        for token in sent:
            if not token.is_stop and not token.is_punct:
                processed_tokens.append(token.text)
        processed_sents.append(" ".join(processed_tokens))

    return nlp(".".join(processed_sents))


def summarize_doc_old(doc, selection_ratio=0.1):
    # Create a list of all the noun chunks in the document
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # Create a dictionary to store the similarity scores between each noun chunk and the whole document
    similarity_scores = {}
    for chunk in noun_chunks:
        chunk = nlp(chunk)
        similarity_scores[chunk.text] = doc.similarity(chunk)

    # Sort the noun chunks by similarity score in descending order
    sorted_chunks = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # Select the top `selection_ratio` of noun chunks to form the summarized document
    num_selected = int(len(noun_chunks) * selection_ratio)
    summarized_chunks = [chunk[0] for chunk in sorted_chunks[:num_selected]]

    return nlp(" ".join(summarized_chunks))


def summarize_text(doc):
    """
    Summarize the given text by selecting the most important sentences.
    """
    # Split the text into sentences and assign a score to each sentence based on its importance
    sentences = [sent for sent in doc.sents]
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_scores[i] = sentence.similarity(doc)

    # Select the most important sentences and sort them by their position in the original text
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:2]
    top_sentences.sort()

    # Combine the top sentences into a summary and return it
    summary = ""
    for i in top_sentences:
        summary += str(sentences[i]) + " "

    summary = summary.strip()
    return summary.strip(), nlp(summary)


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
        if not token.is_alpha or len(token.lemma_) < 3:# allow only 3+-letter words
            continue

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

    top_lemmas_freq = dict(sorted_lemmas[:top])

    top_lemmas_freq_ratio = {}
    top_lemmas = []
    total_doc_len = len(doc)

    for k, v in top_lemmas_freq.items():
        top_lemmas.append(k)
        top_lemmas_freq_ratio[k] = v / total_doc_len

    return top_lemmas_freq_ratio, top_lemmas


def extend_text_with_explanation(input_text, process_data):
    top_lemmas = process_data['general']['top_lemmas']
    bigram_frequency_keys = list(process_data['general']['bigram_frequency'].keys())
    top_subjective_words = list(process_data['persuasion']['top_subjective_words'].keys())
    keywords = process_data['narrative']['keywords']
    positive_top_words = list(process_data['sentiment_analysis']['positive_top_words'].keys())
    negative_top_words = list(process_data['sentiment_analysis']['negative_top_words'].keys())

    # Create a mapping for the words to their corresponding labels
    word_label_map = {}

    for word in top_lemmas:
        word_label_map[word] = "top-lemma"

    for word in top_subjective_words:
        word_label_map[word] = "top-subjective"

    for word in keywords:
        word_label_map[word] = "keyword"

    for word in positive_top_words:
        word_label_map[word] = "top-positive"

    for word in negative_top_words:
        word_label_map[word] = "top-negative"

    # Annotate bigrams
    bigrams = sorted(bigram_frequency_keys, key=len,
                     reverse=True)  # Sort bigrams by length to avoid partial replacements
    for bigram in bigrams:
        input_text = input_text.replace(bigram, f"{bigram} (frequent-combination)")

    # Annotate single words
    words = sorted(word_label_map.keys(), key=len, reverse=True)  # Sort words by length to avoid partial replacements
    for word in words:
        label = word_label_map[word]
        input_text = input_text.replace(word, f"{word} ({label})")

    return input_text
