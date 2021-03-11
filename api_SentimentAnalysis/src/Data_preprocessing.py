# Data_preprocessing.py 
# -- Packages 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import re
import unidecode
from utils import *  # home made package
# NLP package 
import string
from scipy.sparse import save_npz, load_npz # used for saving and loading sparse matrices
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import time


# -- Spacy 
import spacy 
nlp = spacy.load('fr_core_news_md',disable=["tagger","parser", "ner"])
punctuations = string.punctuation

# -- keeep negative words 
stop_words = spacy.lang.fr.stop_words.STOP_WORDS
negative_words =["ne","n'","ni","pas","guère","jamais","plus"]
stop_words = set(stop_words) -set(negative_words)


for word in negative_words:
    nlp.vocab[word].is_stop = False

def Tokenizer_simple(text, accent = False):
    # remove lower case
    lower_text=text.strip().lower()
    # keep negation
    lower_text = lower_text.replace("n'","ne ")
    # remove digit
    lower_text = re.sub('[0-9]', "", lower_text)
    # tokens 
    tokens = word_tokenize(lower_text)
    # remove ponctuations
    table = str.maketrans(punctuations,' '*len(punctuations))
    tokens = [w.translate(table) for w in tokens]
    # tokenize to handle "c'était" like  word 
    tokens = word_tokenize(" ".join(tokens))
    # remove stopword
    tokens= [word.strip() for word in tokens if word not in stop_words and len(word) >= 2 ]
    if accent ==  False:
        p=r"([a-z]{2,})"
        tokens=re.findall(p, unidecode.unidecode(' '.join(tokens)))
    return tokens


def Tokenizer_stemm(text, accent=False):
    # remove lower case
    lower_text=text.strip().lower()
    # keep negation
    lower_text = lower_text.replace("n'","ne ")
    # remove digit
    lower_text = re.sub('[0-9]', "", lower_text)
    # tokens 
    tokens = word_tokenize(lower_text)
    # remove ponctuations
    table = str.maketrans(punctuations,' '*len(punctuations))
    tokens = [w.translate(table) for w in tokens]
    # tokenize to handle "c'était" like  word 
    tokens = word_tokenize(" ".join(tokens))
    # remove stopword
    filtered_text = [word for word in tokens if word not in stop_words and len(word) >=2]
    # stemming
    stemmer  = SnowballStemmer('french')
    tokens = [stemmer.stem(word).strip() for word in filtered_text]
    # remove accent, words >= 1 letters
    if accent == False: 
        p=r"([a-z]{2,})"
        tokens=re.findall(p, unidecode.unidecode(' '.join(tokens)))
    return tokens

def spacyTokenizer_lemma(text, accent=False):
    """
    Tokenizing, token convertion into lowercase, removing stopword and puntuation, and lemmatizing 
    output = Tokens list 
    """
    text = text.strip().lower().replace("n'","ne ")
    tokens = [token.lemma_ for token in nlp(text.lower()) if (not token.is_stop and not token.is_punct and not token.like_num and len(token) >=2)]
    # remove accent
    if accent == False: 
        p=r"([a-z]{2,})"
        tokens = re.findall(p, unidecode.unidecode(' '.join(tokens)))
    return tokens 


def spacyTokenizer_lemma_batch(docs:list):
    norm_docs=[]
    for doc in nlp.pipe(docs, batch_size=32, n_process=3, disable=["tagger","parser", "ner"]):
        norm_docs.append(" ".join([token.lemma_ for token in doc.lower()  if (not token.is_stop and not token.is_punct and not token.like_num and len(token) >1)]))
    return norm_docs


def word2vec_Matrix(X, tokenize_function):
    X_vec = []
    for i, text in enumerate(X):
        if tokenize_function: 
            tokenized_text = " ".join(tokenize_function(text))
            sentence = nlp(tokenized_text.lower())
        else: 
            sentence=nlp(text)
        if sentence.has_vector:
            X_vec.append(sentence.vector)
         # If doc doesn't have a vector, then fill it with zeros.
        else:
            X_vec.append(np.zeros((96,), dtype="float32"))
    return np.array(X_vec)


if __name__=="__main__":
    start_time = time.time()

    # check stop_words
    print(stop_words)

    # -- Directory/file settings
    Dir = os.getcwd()
    model_path = os.path.join(Dir, 'sentiment_pipe.joblib')
    data_path = os.path.join(Dir, 'comments_train.csv')

    # -- Initiatization parameter
    random_state = 0
    test_size = 0.3

    # -- Load Model from disk  
    pipeline = loadModel(model_path)
    print(f'[PIPELINE] {pipeline}')

    # -- Load Data from disk 
    df  = loadData(data_path)
    # -- Data distribution 
    print(f"[DATA] distribution: \n {df.groupby('sentiment').size()} \n")

    # -- Cleaning Text
    ## - Cleaning some text 
    #some_text = df['comment'][1]
    some_text="Je n'ai pas aimé ce restaurant Mauvais Avoir. C'était très bon"
    #some_text="Je n'aime pas . C'est très bien situé mais le service est très Mauvais. Les serveurs ne connaissent pas l'anglais (juste à côté des champions) et ils ne sont pas très serviables. Les fruits de mer sont très frais mais les assiettes ne sont pas aussi salées."
    accent = True
    print(f'Original text: \n {some_text}')
    print(f'Simple token text:  \n {Tokenizer_simple(some_text, accent)}')
    print(f'Stemmed text: \n {Tokenizer_stemm(some_text, accent)}')
    print(f'Lemma text:  \n {spacyTokenizer_lemma(some_text, accent)}')
    print(nlp('bonsoir'.lower()).vector,nlp('bonsoir'.lower()).vector.shape)
    #print([token.lemma_ for token in nlp("était".lower())])

    # df['simple_comment'] = df['comment'].apply(lambda x: Tokenizer_simple(x, accent))
    # df['stemm_comment'] = df['comment'].apply(lambda x: Tokenizer_stemm(x, accent))
    # df['lemma_comment'] = df['comment'].apply(lambda x: spacyTokenizer_lemma(x, accent))
    # # -- Save df to csv 
    # df.to_csv('comments_train_tokenized.csv', index =False, sep=';')

    # print(nlp(some_text).vector,nlp(some_text).vector.shape )
    # print(word2vec(some_text), word2vec(some_text).shape)
    # print(nlp('Bonjour '.lower()).vector)
    # 
    # print(nlp('Bonjour bonsoir'.lower()).vector)
    # print((nlp('Bonjour'.lower()).vector + nlp('bonsoir'.lower()).vector)/2)
    ## -- Cleaning all dataframe 
    ##  - with apply 
    # df['norm_comment'] = df['comment'].apply(lambda text: TextNormalization(text))
    # print(f'Normalized df \n {df.head()}')
    ## - with spacy nlp 
    # docs = df['comment']
    # df[clean_comment] = spacyTokenizer_lemma_batch(docs)

    print("--- %s seconds ---" % (time.time() - start_time))





