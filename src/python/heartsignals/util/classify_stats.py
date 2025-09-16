""" classify_stats.py

    Purpose: Contains a dataclass to handle classifying/training statistics
    Author: Milan Marocchi
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
from transformers import EvalPrediction

def compute_metrics_hf(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    
    # Calculating metrics
    balanced_acc = balanced_accuracy_score(labels, preds)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    mcc_numer = (tp * tn) - (fp * fn)
    mcc_denom = (
        (tp + fp) *
        (tp + fn) *
        (tn + fp) *
        (tn + fn)
    ) ** 0.5

    if mcc_denom <= 1e-8:
        mcc_denom = 1

    sensitivity=tp/sum([tp, fn])
    specificity=tn/sum([tn, fp])
    precision=tp/sum([tp, fp])
    npv=tn/sum([tn, fn])

    mcc = mcc_numer / mcc_denom
    
    # Add other metrics if needed
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    print(tn, fp, fn, tp)
    
    return {
        "balanced_accuracy": balanced_acc,
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "npv": npv,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc
    }


def average_stats(dicts: list[dict]):
    
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)

    for d in dicts:
        for key, value in d.items():
            sum_dict[key] += value
            count_dict[key] += 1

    avg_dict = {key: sum_dict[key] / count for key, count in count_dict.items()}

    return avg_dict

class RunningBinaryConfusionMatrix:
    def __init__(self):
        self.base_stats = defaultdict(int)
        self.loss = 0

        self.aliases = dict(
            loss='Loss',
            acc='Acc',
            tpr='Recall(Sens)',
            tnr='Specif',
            fpr='FPR',
            ppv='Prec',
            npv='NPV',
            f1p='F1+',
            f1n='F1-',
            acc_mu='Mean Acc',
            j='j',
            mcc='MCC',
            qi='qi'
        )

    def update_loss(self, loss):
        self.loss += loss

    def update(self, y_true, y_pred, loss):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        self.base_stats['tn'] += tn
        self.base_stats['fp'] += fp
        self.base_stats['fn'] += fn
        self.base_stats['tp'] += tp

        self.loss += loss

    def total(self):
        return sum([self.base_stats[stat] for stat in self.base_stats])

    def tp(self):
        return self.base_stats['tp']

    def tn(self):
        return self.base_stats['tn']

    def fp(self):
        return self.base_stats['fp']

    def fn(self):
        return self.base_stats['fn']

    def get_stats(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            stats = dict(
                loss=self.loss/self.total(),
                tpr=self.tp()/sum([self.tp(), self.fn()]),
                tnr=self.tn()/sum([self.tn(), self.fp()]),
                fpr=self.fp()/sum([self.fp(), self.tn()]),
                ppv=self.tp()/sum([self.tp(), self.fp()]),
                npv=self.tn()/sum([self.tn(), self.fn()]),
                acc=sum([self.tp(), self.tn()])/self.total(),
            )

            stats.update(dict(  # type: ignore
                acc_mu=np.mean([stats['tpr'], stats['tnr']]),
                qi=np.sqrt(np.prod(stats['tpr']*stats['tnr'])),
                j=sum([stats['tpr'], stats['tnr'], -1]),
                f1p=(2*stats['ppv']*stats['tpr'])/(stats['ppv']+stats['tpr']),
                f1n=(2*stats['npv']*stats['tnr'])/(stats['npv']+stats['tnr']),
            ))

            N = sum([self.tp(), self.tn(), self.fp(), self.fn()])
            S = (self.tp() + self. fn()) / N
            P = (self.tp() + self.fp()) / N
            mcc_numer = (self.tp() / N) - (S * P)
            mcc_denom = (
                P * S * (1 - S) * (1 - P)
            ) ** 0.5

            if mcc_denom <= 1e-8:
                mcc_denom = 1

            mcc = mcc_numer / mcc_denom
            stats.update(dict(  # type: ignore
                mcc=mcc,
            ))

        return stats

    def get_loss(self):
        return {"loss": self.loss}

    def display_stats(self, aliases=True):
        if aliases is True:
            aliases = self.aliases
        else:
            aliases = {}

        stats = self.get_stats()

        for a in stats:
            if a not in aliases:
                aliases[a] = a

        formatted_stats = ', '.join(
            f'{aliases[stat]}: {stats[stat]:.3f}' for stat in aliases)
        return formatted_stats
