import spacy

from spacytextblob.spacytextblob import SpacyTextBlob

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

nlp.add_pipe('spacytextblob')
