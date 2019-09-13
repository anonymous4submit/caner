import os
import logging
from pathlib import Path
from caner.domain_pipeline import get_org_dict
import pandas as pd
logger = logging.getLogger(__name__)


def get_ner_metrics(source_file, real_file, predict_file, org_domain=None):
    """
    Calculate and print NER metrics (P, R, F)
    :param source_file: The source file to be predicted
    :param real_file: The real target file of above source_file
    :param predict_file: The model prediction file of above source_file
    :param org_domain: If None, include all domain, else only include specific domain
    """
    org_dict = get_org_dict()
    with open(predict_file, 'r', encoding='utf-8') as f:
        predict_sentence_list = f.readlines()
    with open(real_file, 'r', encoding='utf-8') as f:
        real_sentence_list = f.readlines()
    with open(source_file, 'r', encoding='utf-8') as f:
        source_sentence_list = f.readlines()
    assert len(predict_sentence_list) == len(real_sentence_list) == len(source_sentence_list)
    match_num = p_denom = r_denom = 0.
    """
    The domain of the denominator of the precision is uncertain, 
    so it is approximated by the recent history. (For experimental reference only)
    """
    history_domain = ''
    for i in range(len(predict_sentence_list)):
        pred_list = predict_sentence_list[i].split()
        real_list = real_sentence_list[i].split()
        source_list = source_sentence_list[i].split()
        pred_list.append('O')
        real_list.append('O')
        source_list.append('O')
        assert len(pred_list) == len(real_list) == len(source_list)

        for j in range(len(pred_list)):
            pred_org = ''
            real_org = ''
            if real_list[j][0] == 'B':
                real_org = source_list[j]
                for k in range(j+1, len(pred_list)):
                    if real_list[k][0] == 'I':
                        real_org += source_list[k]
                    else:
                        break
                # history_domain = org_dict[real_org]
                if org_domain is None or org_dict[real_org] == org_domain:
                    r_denom += 1
            if pred_list[j][0] == 'B':
                pred_org = source_list[j]
                for k in range(j+1, len(pred_list)):
                    if pred_list[k][0] == 'I':
                        pred_org += source_list[k]
                    else:
                        break
                # if org_domain is None or history_domain == org_domain:
                if True:
                    p_denom += 1
            if pred_list[j] == real_list[j] and pred_list[j][0]=='B' and pred_org == real_org:
                if org_domain is None or org_dict[real_org] == org_domain:
                    match_num += 1

    precision = match_num / p_denom
    recall = match_num / r_denom

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = precision * recall * 2 / (precision + recall)
    if org_domain is None:
        org_domain = 'None'

    print('Domain: %s, Match: %d, P_denom: %d, R_denom: %d, Precision: %.6f, Recall: %.6f, F1-Score: %.6f' %
          (org_domain, int(match_num), int(p_denom), int(r_denom), precision, recall, f1))


def get_ner_metrics_2(real_file, predict_file):
    """
    For experimentation, please ignore it.
    """
    with open(predict_file, 'r', encoding='utf-8') as f:
        predict_sentence_list = f.readlines()
    with open(real_file, 'r', encoding='utf-8') as f:
        real_sentence_list = f.readlines()
    assert len(predict_sentence_list) == len(real_sentence_list)
    match_num = p_denom = r_denom = 0.

    for i in range(len(predict_sentence_list)):
        pred_list = predict_sentence_list[i].split()
        real_list = real_sentence_list[i].split()

        assert len(pred_list) == len(real_list)
        for p, r in zip(pred_list, real_list):
            if p == r == 'O':
                continue
            if p != 'O' and r != 'O':
                match_num += 1
            if p != 'O':
                p_denom += 1
            if r != 'O':
                r_denom += 1

    precision = match_num / p_denom
    recall = match_num / r_denom
    f1 = precision * recall * 2 / (precision + recall)

    print('Match: %d, P_denom: %d, R_denom: %d, Precision: %.6f, Recall: %.6f, F1-Score: %.6f' %
          (int(match_num), int(p_denom), int(r_denom), precision, recall, f1))
