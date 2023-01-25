# Fake Curator Process API
TODO describe

## Flow
- request from UI to process
- read Firestore, throw if Article missing
- process
- update Database with new data - mark as "processed: true"
- return to UI full doc

Process:
Process contains not only of default NLP/Sentiment stuff, but also figuring out % of how many of what exists
Later it would be multiplied and calculated by CoefWeigts - determiates improtance of each part of procssed data.