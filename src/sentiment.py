from collections import Counter


def analyze_sentiment(doc):
    sentiment_scores = []

    # Initialize counters for positive, neutral, and negative sentiment
    pos_count = 0
    neu_count = 0
    neg_count = 0

    # Initialize lists to store the words for each sentiment
    pos_words = []
    neu_words = []
    neg_words = []

    # Loop through each token in the document
    for token in doc:
        if not token.is_alpha:
            continue

        sentiment_scores.append(token._.blob.polarity)

        # Check if the token has a sentiment score
        if token._.blob.polarity > 0:
            pos_count += 1
            pos_words.append(token.text)
        elif token._.blob.polarity == 0:
            neu_count += 1
            neu_words.append(token.text)
        else:
            neg_count += 1
            neg_words.append(token.text)

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    # Calculate the overall sentiment as a percentage of positive, neutral, and negative words
    total = pos_count + neu_count + neg_count
    pos_ratio = pos_count / total
    neu_ratio = neu_count / total
    neg_ratio = neg_count / total

    # Get the most frequently used words for each sentiment
    pos_top_words = dict(Counter(pos_words).most_common(10))
    neu_top_words = dict(Counter(neu_words).most_common(10))
    neg_top_words = dict(Counter(neg_words).most_common(10))

    # Return the results
    return {
        "avg_sentiment": avg_sentiment,
        "positive_ratio": pos_ratio,
        "positive_top_words": pos_top_words,
        "neutral_ratio": neu_ratio,
        "neutral_top_words": neu_top_words,
        "negative_ratio": neg_ratio,
        "negative_top_words": neg_top_words,
    }
