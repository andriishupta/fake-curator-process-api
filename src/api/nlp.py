import spacy

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")