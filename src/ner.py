def get_ner_frequency(doc):
    # Calculate the total number of entities
    num_entities = sum(1 for ent in doc.ents)

    # Create a dictionary to store the entity counts
    entity_counts = {}

    # Loop through the entities and update the counts
    for ent in doc.ents:
        entity_text = ent.text.lower()
        if entity_text in entity_counts:
            entity_counts[entity_text] += 1
        else:
            entity_counts[entity_text] = 1

    # Sort the entity counts in descending order
    sorted_entity_counts = {k: v for k, v in sorted(entity_counts.items(), key=lambda item: item[1], reverse=True)}

    # Calculate the number of entities in the top 10%
    num_top_entities = int(num_entities * 0.1)

    # Create a new dictionary to store the top entities and their counts
    top_entity_counts = {}

    # Loop through the sorted entity counts and add the top entities to the new dictionary
    for entity_text, count in sorted_entity_counts.items():
        if len(top_entity_counts) >= num_top_entities:
            break
        top_entity_counts[entity_text] = count

    return top_entity_counts
