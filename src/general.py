from heapq import nlargest

from api.nlp import nlp


def filter_doc(doc):
    filtered_tokens = []

    # Iterate over the tokens in the processed text
    for token in doc:
        # Check if the token is a stop word, punctuation or contains some numbers/symbols
        if not (token.is_stop or token.is_punct or token.is_alpha):
            # If not, add the token to the filtered list
            filtered_tokens.append(token)

    body_clean_text = " ".join([token.text for token in filtered_tokens])

    return nlp(body_clean_text)


def doc_summarization(doc, ratio: float = 0.1):
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
    summarized_sentences = nlargest(int(len(sentences) * ratio), sentence_scores, key=sentence_scores.get)

    return summarized_sentences
