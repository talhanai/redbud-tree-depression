# redbud-tree-depression
This repo contains scripts to model depression in speech and text. LSTM models are utilized to model at the segment-level of an interview (not at the word-level). The two modalities are also combined and fed into a feedforward network. 

### Data
The data used can be downloaded from the [Distress Analysis Interview Corpus](http://dcapswoz.ict.usc.edu/), and contains audio, video, and text of interviews with 189 subjects, about 20% of whom had some level of depression.

### Features
The features are either segment-level statistics of the audio, or doc2vec embeddings of the words in a segment. Higher-level audio features (mean, max, min, median, std) were extracted using the COVAREP and FORMNAT features provided in the corpus, and the doc2vec embeddings were generated using [this script](https://github.com/talhanai/sweet-wrapper-embeddings). I trained using the binary outcomes as well as the multi-class outcomes.

### Files
The repo contains the following files:

- **trainLSTM.py** which contains the methods used to train the models.
- **requirements.txt** which are the libraries used in the conda environment of this project.

Keras with the tensorflow back-end was used for modeling.

Interested in using my audio/text features? [Let me know](mailto:tuka@mit.edu).

## Libraries
I used the following librarires:
```
keras-gpu=2.1.3=py36_0
cudnn=7.0.5=cuda8.0_0
tensorflow-gpu=1.4.1=0
tensorflow-gpu-base=1.4.1=py36h01caf0a_0
tensorflow-tensorboard=0.4.0=py36hf484d3e_0
```

## Reference Paper

``` 
T. Alhanai, MM. Ghassemi, J. Glass, 
"Detecting "Detecting Depression with Audio/Text Sequence Modeling of Interviews"
Interspeech 2018, India
```



DISCLAIMER: The user accepts the code / configuration / repo AS IS, WITH ALL FAULTS.
