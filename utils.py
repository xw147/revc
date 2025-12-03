from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score

import numpy as np

def __semitest__(base_classifier_func, pred, outc, label_len, job_indx = 1):
    row_indx = np.random.permutation(list(range(pred.shape[0])))
    pred = pred[row_indx,:]
    outc = outc[row_indx]

    labeled_pred = pred[:label_len,:]
    unlabeled_pred = pred[label_len:,:]
    labeled_outc = outc[:label_len]
    unlabeled_outc = outc[label_len:]

    # Create fresh classifier instance for each call
    base_estimator = base_classifier_func()
    base_estimator.fit(labeled_pred, labeled_outc)
    base_ests = base_estimator.predict(unlabeled_pred)
    base_acc = accuracy_score(unlabeled_outc, base_ests)

    k_heuristic = 50 # tunable hyperparameter
    # Create another fresh classifier for semi-supervised learning
    semi_base = base_classifier_func()
    semi_estimator = SelfTrainingClassifier(semi_base, criterion = 'k_best', k_best = k_heuristic)
    semi_estimator.fit(pred, labeled_outc.tolist() + [-1]*len(unlabeled_outc))
    semi_ests = semi_estimator.predict(unlabeled_pred)
    semi_acc = accuracy_score(unlabeled_outc, semi_ests)

    return [base_acc, semi_acc]