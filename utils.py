import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score

def metrics_graph(yt, yp, threshold=None):
    """Evaluate DTI predictions with AUC, AUPR, F1, and accuracy.

    AUC/AUPR are threshold-free. F1 and accuracy use a decision threshold:
    if ``threshold`` is None it is chosen by max-F1 on *this* set (use on
    validation only); otherwise that frozen threshold is applied (use on test).

    Returns:
        auc, aupr, f1, accuracy, threshold
    """
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)

    if threshold is not None:
        pred = (yp >= float(threshold)).astype(np.float64)
        tp = np.sum((pred == 1) & (yt == 1))
        fp = np.sum((pred == 1) & (yt == 0))
        fn = np.sum((pred == 0) & (yt == 1))
        tn = np.sum((pred == 0) & (yt == 0))
        f1 = float(2 * tp / (2 * tp + fp + fn + 1e-12))
        acc = float((tp + tn) / max(len(yt), 1))
        return auc, aupr, f1, acc, float(threshold)

    real_score=np.mat(yt)
    predict_score=np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    chosen_threshold = float(thresholds[0, max_index])
    return auc, aupr, f1_score[0, 0], accuracy[0, 0], chosen_threshold


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int) -> None:
    """Adjust learning rate exponentially every 50 epochs."""
    lr = 0.001 * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility across torch, numpy, and random libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
