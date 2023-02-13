import spacy

from spacytextblob.spacytextblob import SpacyTextBlob

spacy.cli.download("en_core_web_md")
nlp = spacy.load("en_core_web_md")

nlp.add_pipe('spacytextblob')
