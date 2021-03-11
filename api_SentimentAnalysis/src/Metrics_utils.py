# Metrics_utils.py

# -- Packages 

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
import matplotlib.pyplot as plt




# -- NLP 
import spacy
nlp = spacy.load('fr_core_news_md')

def KfoldCrossValidation(model, X_train,y_train, cv):
    scoring = {'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1_score': 'f1',
                'auc':'roc_auc',
                }
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True, verbose=1)
    # print(scores.keys()) # => (['fit_time', 'score_time', 'test_accuracy', 'train_accuracy', 'test_precision', 'train_precision', 'test_recall', 'train_recall', 'test_f1_score', 'train_f1_score', 'test_auc', 'train_auc'])
    print(f'\n[METRICS] {model} :')
    for score in list(scores)[2:]:
        print(f"Avg {score}: {np.mean(scores[score]):.3f} | sdt (+/- {np.std(scores[score]):.2f})" )
    return {'CV_train_score': 100 * np.mean(scores['train_accuracy']).round(3) , 'eval_score': 100 *np.mean(scores['test_accuracy']).round(3) ,'eval_precision': np.mean(scores['test_precision']).round(3) ,'eval_recall':np.mean(scores['test_recall']).round(3) ,'eval_F1score':np.mean(scores['test_f1_score']).round(3) ,'eval_AUC':np.mean(scores['test_auc']).round(3) }


def runGridSearchCV(model, X_train,y_train,params,cv):
    scoring = {
    'f1_score':  make_scorer(f1_score),
    'roc_auc': 'roc_auc'
    }
    grid = GridSearchCV(model, 
                param_grid=params, 
                scoring=scoring, 
                verbose=1, 
                n_jobs=3, 
                cv=cv,
                return_train_score=True,
                refit='f1_score'
            )
    grid_results = grid.fit(X_train, y_train)
    print('Best Score: ', grid_results.best_score_)
    print('Best Params: ', grid_results.best_params_)
    print(sorted(grid_results.cv_results_.keys()))
    return {'cv_results': grid_results.cv_results_, 'Best_score':grid_results.best_score_, 'Best_params': grid_results.best_params_}


def ClassificationMetrics(model,X_train,y_train,X_test,y_test):
    """
    Different scores to measure binary classification performance. 
    Some metrics might require probability estimates of the positive class sample.

    Input Parameters
    ----------------
    model : Fitted model used.
    X_train: Training data.
    y_train: Ground truth (correct) training labels.
    X_test:  Test data.
    y_train: Ground truth (correct) test labels

    Returns
    ---------------
    Metric scores dictionnary 
    """

    y_pred= model.predict(X_test)
    try: 
        y_proba= model.predict_proba(X_test)[:, 1]
        auc_proba = roc_auc_score(y_test, y_proba)
        fp_rate_proba, tp_rate_proba, thresholds_proba = roc_curve(y_test, y_proba)
    except AttributeError as e:
        print(e)
        auc_proba = None
        # model['clf'].set_params(probability=True)
        # model.fit(X_train,y_train)
        # y_proba = model.predict_proba(X_test)[:, 1]
    train_score= model.score(X_train, y_train) 
    test_score = model.score(X_test, y_test)
    precision, recall, f1score, support = score(y_test, y_pred,  average='binary')
    conf_matrix = confusion_matrix(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    AUC = roc_auc_score(y_test, y_pred).round(3) 
    fp_rate, tp_rate, thresholds = roc_curve(y_test, y_pred)

    print(f'\n[METRICS] {model} :')
    print("Train score: {0:.2f} %".format(100 * train_score))
    print("Test score: {0:.2f} %".format(100 * test_score))
    print('precision: {0:.2f}'.format(precision))
    print('recall: {0:.2f}'.format(recall))
    print('f1score: {0:.2f}'.format(f1score))
    print('support: {}'.format(support))
    print('Classification Report: \n {}'.format(report))
    print('AUC: %.3f' % AUC)
    if auc_proba:
        print('AUC proba: %.3f' % auc_proba)
    print('confusion Matrix: \n {} \n'.format(conf_matrix))
    return {'train_score': 100 * train_score.round(3) , 'test_score': 100 * test_score.round(3) ,'precision': precision.round(3) ,'recall':recall.round(3) ,'F1score':f1score.round(3) ,'AUC':AUC.round(3) ,'AUC_proba':auc_proba }

def ROCcurve(model,model_name:str, X_test,y_test):
    """
    Compute Receiver operating characteristic (ROC).
    Note: this implementation is restricted to the binary classification task.

    Input Parameters
    ----------------
    model : Fitted model used.
    model_name: model name 
    X_test:  Test data.
    y_train: Ground truth (correct) test labels

    Return
    ---------------
    ROC curve plot
    """
    try:
        y_probs = model.predict_proba(X_test)
            # keep probabilities for the positive outcome only
        y_proba = y_probs[:, 1]
    except:
        y_proba = model.predict(X_test)
    # calculate scores
    rc_proba = [0 for _ in range(len(y_test))]
    rc_auc = roc_auc_score(y_test, rc_proba)
    auc_proba = roc_auc_score(y_test, y_proba)
    # summarize scores
    #print('RC : ROC AUC=%.3f' % (rc_auc))
    #print('Model: ROC AUC=%.3f' % (auc_proba))
    # calculate roc curves
    rc_fpr, rc_tpr, _ = roc_curve(y_test, rc_proba)
    fp_rate, tp_rate, thresholds = roc_curve(y_test, y_proba)
    # plot the roc curve for the model
    plt.plot(rc_fpr, rc_tpr, linestyle='--') #  label='Random Chances'
    plt.plot(fp_rate, tp_rate, label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
        


def plotConfusionMatrix(model,model_name, X_test,y_test, normalize=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    
    Input Parameters
    ----------------
    model: Fitted model used.
    model_name: model name 
    X_test:  Test data.
    y_train: Ground truth (correct) test labels
    Normalize: Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. If None, confusion matrix will not be normalized.

    Return
    ---------------
    ROC curve plot
    """
    matrix = plot_confusion_matrix(model, X_test, y_test,
                                    cmap=plt.cm.Blues,
                                    normalize=normalize,
                                )
                                    
    plt.title(f'Confusion matrix for {model_name}')
    plt.show(matrix)
    plt.show()


def SaveConfusionMatrix(model, model_name, X_test, y_test, output_path,  normalize=None):
    """
    Compute and save confusion matrix
    """
    fig = plot_confusion_matrix(model, X_test, y_test,
                                    cmap=plt.cm.Blues,
                                    normalize=normalize,
                                    display_labels=['Negatif','Positif']
                                    )
    if normalize == 'true':
        plt.title(f'Normalized confusion matrix for {model_name}')
    else:
        plt.title(f'Confusion matrix for {model_name}')
    plt.savefig(output_path, bbox_inches='tight')  
    plt.close()


def plotGridSearchCVscorers(results, param,output_path):
    """ 
    Plotting results of a GridSearchCV run using f1_score and rox_auc evaluation metrics

    Input Parameters
    ----------------
    results: GridsearchCV metric results
    param :  estimator  parameter that has been tuned
    output_path : path of the saved figure 

    output
    ------
    Saved figure 
    """

    scoring = {'f1_score':  make_scorer(f1_score), 'roc_auc': 'roc_auc'}

    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)
    plt.xlabel(param)
    plt.ylabel("Score")
    ax = plt.gca()

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results[param].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()
    plt.savefig(output_path, bbox_inches='tight')  
    plt.close()


class WordVectorTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, tokenizer= None, model='fr_core_news_md'):
        self.model = model
        self.tokenizer = tokenizer
    def fit(self, X,y=None):
        return self
    def transform(self,X):
        nlp= spacy.load(self.model)
        # if self.tokenizer is not None:
        #     X = self.tokenizer(X)
        #     print(f'[TYPE] {type(X)}')
        return np.concatenate([nlp(doc).vector.reshape(1,-1) for doc in np.array(X)])

def spacy_word2vec_Matrix(df, text_col, nlp = spacy.load('fr_core_news_md')):
    with nlp.disable_pipes():
        comment_vectors = np.array([nlp(text).vector for text in list(df[text_col].astype('str').values) ])
    print(f'[Comment Vector Matrix] {comment_vectors.shape}')

