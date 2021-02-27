
# Analyse de sentiments 
## Simplon | Mise en situation | CasPratique_1

### Projet 
Je dispose du code d’une application Flask, écrit par un collègue, dont le but est d’effectuer une analyse de sentiments (positif/négatif) d’un texte fourni par un utilisateur. Pour réaliser les prédictions, le code fait appel à un modèle sauvegardé dans un fichier joblib (sentiment_pipe.joblib) dont je ne connais en rien le contenu.

Mon objectif est de restructurer et d’améliorer le code ainsi que le modèle d’analyse de sentiment, pour le rendre plus fonctionnel, performant et réutilisable par mes collègues sans aucun bugs.

### Les données
Il s’agit d’un corpus (fichier “comments_train.csv”) de 1617 avis/commentaires d’utilisateurs sur des restaurants en Français :
Il est composé d’une colonne “comment” contenant les commentaires et d’une colonne “sentiment” indiquant si le commentaire est négatif ou positif.

Distribution des avis: négatif 598 (37%) | positif 1019 (63%)

### Pipelines testées --> **script : models.py**
#### Pré-traitement du texte
- Aucun prétraitement  (“No tokenizer”)
- Tokenisation (“Simple tokenierr”)
- Tokenisation + stemming (“Stemm tokenizer”)
- Tokenisation + lemmatisation (“Spacy lemma tokenizer”)

#### Vectorisation 
- Countvectorizer (unigram ou bigram) 
- TF-IDF (unigram ou bigram) 
- Word Embedding: Glove (Spacy), Word2vec, Fastext

#### Classification
- Logistic regression 
- Naive Baye 
- Random Forest 
- SGDclassifier 
- SVM (linear et rbf)

Pour determiner le meilleure modèle : StratifiedKFold Cross Validation
Pour determiner les meilleurs hyperparamètres paramètres : GridsearchCV --> **script : model_GridSearchCV.py**

### Deep Learning
- Language Model : Transfer-learning avec CamemBERT 
