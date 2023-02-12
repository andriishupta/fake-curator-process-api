def filter_stopwords(nlp_doc):
    filtered_tokens = []

    # Iterate over the tokens in the processed text
    for token in nlp_doc:
        # Check if the token is a stop word
        if not token.is_stop:
            # If not, add the token to the filtered list
            filtered_tokens.append(token)

    return filtered_tokens
