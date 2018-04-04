# redbud-tree-depression
This repo contains scripts to model depression in speech and text. LSTM models are utilized to model at the segment-level of an interview (i.e. not at the word-level). The features are either segment-level statistics of the audio, or doc2vec embeddings of the words in a segment.

The data used can be downloaded from the [Distress Analysis Interview Corpus](http://dcapswoz.ict.usc.edu/).

The repo contains the following files:

- **trainLSTM.py** which contains the methods used to train the models.
- **requirements.txt** which are the libraries used in the conda environment of this project.
