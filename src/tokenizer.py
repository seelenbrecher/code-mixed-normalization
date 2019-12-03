import numpy as np
import random
import sklearn_crfsuite
import scipy

from importlib import reload
from collections import Counter
from sklearn.metrics import classification_report, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

"""
Tokenizer module uses sequence labelling at character level.
Using three tags, "B": Beginning of the word, "I": Inside the word, "O": Not included in any word
"""
class Tokenizer:
    model = None
    
    # parameter
    window = 5
    c1 = 0.1
    c2 = 0.1

    def __init__(self):
        random.seed(0)
        self.model = sklearn_crfsuite.CRF(
                        algorithm='lbfgs',
                        max_iterations=100,
                        all_possible_transitions=True,
                        c1=1.42023576422621,
                        c2=0.0467142909420063,
                    )
    
    """
    Using several features:
    1. char: current character
    2. is_alpha: true if current character is alpha
    3. is_numeric: true if current character is numeric
    4. other: true if current character neither alpha nor numeric
    5. is_upper: true if curr character is capital
    6. char[X]: N-window character from current character
    """
    def _extract_features(self, tweet):
        features = []
        for i in range(0, len(tweet)):
            char = tweet[i]

            feature = {
                'char': char,
                'is_alpha': char.isalpha(),
                'is_numeric': char.isnumeric(),
                'other': not char.isalpha() and not char.isnumeric(),
            }

            if char.isalpha():
                feature['is_upper'] = char.isupper()

            for j in range(1, self.window + 1):
                idx = i - j
                if idx < 0:
                    continue
                feature[f'char[{-j}]'] = tweet[idx]
            
            for j in range(1, self.window + 1):
                idx = i + j
                if idx >= len(tweet):
                    continue
                feature[f'char[{j}]'] = tweet[idx]

            features.append(feature)
        
        return features
    
    def _extract_labels(self, encoded_label):
        labels = []
        for i in range(0, len(encoded_label)):
            labels.append(encoded_label[i])
        return labels
    
    def _get_contexts_from_encoded_label(self, text, labels):
        contexts = []
        context = ''
        for char, label in zip(text, labels):
            if label == 'O':
                continue
            if label == 'B':
                if context != '':
                    contexts.append(context)
                    context = ''
            context += char
        if context != '':
            contexts.append(context)
        return contexts
    
    def train_hyper_param_optimization(self, X, y):
        X_train = []
        y_train = []

        for tweet, encoded_label in zip(X, y):
            X_train.append(self._extract_features(tweet))
            y_train.append(self._extract_labels(encoded_label))
        
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted',
                                labels=['B', 'I', 'O'])

        rs = RandomizedSearchCV(self.model, params_space,
                        cv=4,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
        
        rs.fit(X_train, y_train)

        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        self.c1 = rs.best_params_['c1']
        self.c2 = rs.best_params_['c2']
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True,
            c1=self.c1,
            c2=self.c2,
        )

    """
    X will be tweet
    y will be encoded label
    """
    def train(self, X, y):
        X_train = []
        y_train = []

        for tweet, encoded_label in zip(X, y):
            X_train.append(self._extract_features(tweet))
            y_train.append(self._extract_labels(encoded_label))

        self.model.fit(X_train, y_train)
    
    """
    @params list of tweet
    @return encoding text and list of contexts
    """
    def predict(self, X):
        features = []
        for tweet in X:
            features.append(self._extract_features(tweet))

        y_preds = self.model.predict(features)

        contexts = []
        for tweet, y_pred in zip(X, y_preds):
            contexts.append(self._get_contexts_from_encoded_label(tweet, y_pred))
        
        return y_preds, contexts
    
    """
    accuracy encoded label
    """
    def accuracy(self, y, y_preds):
        labels = {'B': 0, 'I': 1, 'O': 2}
        predictions = np.array([labels[tag] for row in y_preds for tag in row])
        truths = np.array([labels[tag] for row in y for tag in row])

        report = classification_report(
            truths, predictions,
            target_names=['B', 'I', 'O'],
        )

        print(report)

        print('confussion matrix')
        flat_y = [item for y_ in y for item in y_]
        flat_y_preds = [item for y_pred in y_preds for item in y_pred]
        conf_mat = confusion_matrix(flat_y, flat_y_preds, labels=['B', 'I', 'O'])
        print(conf_mat)

        report = classification_report(
            truths, predictions,
            target_names=['B', 'I', 'O'],
            output_dict=True,
        )
        return report
    
    def accuracy_context(self, y, y_preds):
        precision = 0
        recall = 0
        exact = 0
        for y_, y_pred in zip(y, y_preds):
            lcs_, same_context = lcs(y_, y_pred)
            precision += lcs_ / len(y_)
            recall += lcs_ / len(y_pred)
        precision /= len(y)
        recall /= len(y)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
