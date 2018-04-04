# redbud-tree-depression
This repo contains scripts to model depression in speech and text. LSTM models are utilized to model at the segment-level of an interview (not at the word-level). The two modalities are also combined and fed into a feedforward network. 

### Data
The data used can be downloaded from the [Distress Analysis Interview Corpus](http://dcapswoz.ict.usc.edu/), and contains audio video, and text of interviews with 189 subjects, about 20% of whom had some level of depression.

### Features
The features are either segment-level statistics of the audio, or doc2vec embeddings of the words in a segment. Higher-level audio features (mean, max, min, median, std) were extracted using the COVAREP and FORMNAT features provided in the corpus, and the doc2vec embeddings were generated using [this script](https://github.com/talhanai/sweet-wrapper-embeddings). I trained using the binary outcomes as well as the multi-class outcomes.

### Files
The repo contains the following files:

- **trainLSTM.py** which contains the methods used to train the models.
- **requirements.txt** which are the libraries used in the conda environment of this project.

Keras with the tensorflow back-end was used for modeling.

Interested in using my audio/text features? [Let me know](mailto:tuka@mit.edu).
