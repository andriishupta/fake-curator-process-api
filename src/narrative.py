from api.nlp import nlp


def extract_keywords(lemma_frequency, selection_ratio=0.1):
    lemma_count = sum(lemma_frequency.values())

    keywords = []
    for lemma, count in lemma_frequency.items():
        lex_div = count / lemma_count
        if lex_div >= 0.1 and nlp.vocab[lemma].is_oov == False:
            keywords.append((lemma, count))

    # Sort the keywords by frequency and return only the top 10% of the keywords
    keywords.sort(key=lambda x: x[1], reverse=True)
    return keywords[:int(selection_ratio * len(keywords))]


def detect_entity_frequency(doc, selection_ratio=0.1):
    entities = [ent.text for ent in doc.ents]
    entity_frequency = {}
    for entity in entities:
        if entity in entity_frequency:
            entity_frequency[entity] += 1
        else:
            entity_frequency[entity] = 1
    sorted_entity_frequency = sorted(entity_frequency.items(), key=lambda x: x[1], reverse=True)
    top_entities_count = int(len(sorted_entity_frequency) * selection_ratio)
    top_entities = sorted_entity_frequency[:top_entities_count]

    return top_entities


def detect_entity_salience(doc, selection_ratio=0.1):
    entities = [(ent.text, ent.sent_start, ent.sent_end, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    entity_salience = {}
    for entity in entities:
        entity_text = entity[0]
        entity_sent_start = entity[1]
        entity_sent_end = entity[2]
        if entity_text in entity_salience:
            entity_salience[entity_text] += (entity_sent_end - entity_sent_start) / len(doc.sents)
        else:
            entity_salience[entity_text] = (entity_sent_end - entity_sent_start) / len(doc.sents)
    sorted_entity_salience = sorted(entity_salience.items(), key=lambda x: x[1], reverse=True)
    top_entities_count = int(len(sorted_entity_salience) * selection_ratio)
    top_entities = sorted_entity_salience[:top_entities_count]

    return top_entities
