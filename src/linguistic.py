def detect_unusual_inappropriate_language_ratio(doc):
    unusual_inappropriate_count = 0

    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue

        if token.is_alpha and (not token.is_oov):
            continue

        unusual_inappropriate_count += 1

    if unusual_inappropriate_count == 0:
        return 0

    return unusual_inappropriate_count / len(doc)


def detect_awkward_text_ratio(doc):
    awkward_tokens = 0
    for token in doc:
        if (
                token.dep_ in ("amod", "compound", "nsubj", "dobj", "pobj") and
                not token.is_stop and
                not token.is_punct
        ):
            awkward_tokens += 1
    return awkward_tokens / len(doc)
