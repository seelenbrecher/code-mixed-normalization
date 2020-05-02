import numpy as np
import random
import sklearn_crfsuite
import re

from sklearn.metrics import classification_report, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split

class LanguageIdentifier:
    model = None
    
    # parameter
    window = 5
    c1 = 0.1
    c2 = 0.1

    def __init__(self, dict_id=None, dict_en=None):
        random.seed(0)
        self.model = sklearn_crfsuite.CRF(
                        algorithm='lbfgs',
                        max_iterations=100,
                        all_possible_transitions=True,
                        c1=0.0012408441118607547,
                        c2=0.04530982470537112,
                    )
    
    def _extract_features(self, words):
        features = []

        for idx, word in enumerate(words):
            feature = {
                'n_gram_0': word,
                'is_alpha': word.isalpha(),
                'is_numeric': word.isnumeric(),
                'is_capital': word.isupper(),
                'contains_alpha': bool(re.search('[a-zA-Z]', word)),
                'contains_numeric': bool(re.search('[0-9]', word)),
                'has_aphostrophe': '\'' in word,
            }
            # if idx != 0:
            #     feature['prev_word'] = words[idx-1]
            # if idx != len(words) - 1:
            #     feature['next_word'] = words[idx+1]
            if len(word) > 5:
                for i in range(1, len(word) - self.window):
                    feature[f'n_gram_{i}'] = word[i:(i+self.window)]
            
            features.append(feature)

        return features
    
    def train_hyper_param_optimization(self, X, y):
        X_train = []
        y_train = y

        for tweet, encoded_label in zip(X, y):
            X_train.append(self._extract_features(tweet))
        
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted',
                                labels=['en', 'id', 'un'])

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
    X = list of words. E.g X = [['I', 'love', 'mamaku', ':)']]
    y = list of language. E.g y = [['en', 'en', 'id', 'un']]
    """
    def train(self, X, y):
        X_train = []

        for words in X:
            X_train.append(self._extract_features(words))

        self.model.fit(X_train, y)

    """
    @params list of tweet
    @return encoding text and list of contexts
    """
    def predict(self, X):
        features = []
        for words in X:
            features.append(self._extract_features(words))
        
        y_preds = self.model.predict(features)
        
        return y_preds
    
    def accuracy(self, y, y_preds):
        labels = {'en': 0, 'id': 1, 'un': 2}
        predictions = np.array([labels[tag] for row in y_preds for tag in row])
        truths = np.array([labels[tag] for row in y for tag in row])

        report = classification_report(
            truths, predictions,
            target_names=['en', 'id', 'un']
        )

        print(report)

        print('confussion matrix')
        flat_y = [item for y_ in y for item in y_]
        flat_y_preds = [item for y_pred in y_preds for item in y_pred]
        conf_mat = confusion_matrix(flat_y, flat_y_preds, labels=['en', 'id', 'un'])
        print(conf_mat)

        report = classification_report(
            truths, predictions,
            target_names=['en', 'id', 'un'],
            output_dict=True,
        )
        accuracy = 0
        for y, y_pred in zip(flat_y, flat_y_preds):
            if y == y_pred:
                accuracy += 1
        accuracy /= len(flat_y)
        report['accuracy'] = accuracy
        return report