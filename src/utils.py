import math
from collections import defaultdict

import torch
from nltk.translate.bleu_score import corpus_bleu
from torch import nn

from eval.bleu import corpus_bleu
from eval.rouge import Rouge
# from eval.meteor import Meteor
from DECOM_model import sequence_mask


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_seq2seq_decoder`"""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len, average=True):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss.mean() if average else weighted_loss.sum()


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class DAMSMLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(DAMSMLoss, self).__init__()
        self.alpha = alpha
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, anchor, positive, negative):
        pos_sim = self.cos(anchor, positive).unsqueeze(1)
        neg_mat = sim_matrix(anchor, negative)

        logit = torch.exp(torch.div(neg_mat - pos_sim, self.alpha))
        logit = torch.log(1 + torch.sum(logit, dim=1))

        return logit.mean()


def eval_bleu_rouge_meteor(ids, comment_pred, comment):
    """An unofficial evalutation helper.
     Arguments:
        ids: list: list of id for the reference comments
        comment_pred: list: list of tokens for the prediction comments
        comment: list: list of tokens for the reference comments
    """
    assert len(ids) == len(comment_pred) == len(comment)

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, _ = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(ids, comment_pred, comment)
    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(ids, comment_pred, comment)

    # # Compute METEOR scores
    # meteor_calculator = Meteor()
    # meteor, _ = meteor_calculator.compute_score(ids, comment_pred, comment)

    return bleu * 100, rouge_l * 100, -1 * 100, ind_bleu, ind_rouge


# implement by "Towards automatically generating block comments for code snippets"
def score_sentence(pred, gold, ngrams, smooth=1e-5):
    scores = []
    # Get ngrams count for gold.
    count_gold = defaultdict(int)
    _update_ngrams_count(gold, ngrams, count_gold)
    # Init ngrams count for pred to 0.
    count_pred = defaultdict(int)
    # p[n][0] stores the number of overlapped n-grams.
    # p[n][1] is total # of n-grams in pred.
    p = []
    for n in range(ngrams + 1):
        p.append([0, 0])
    for i in range(len(pred)):
        for n in range(1, ngrams + 1):
            if i - n + 1 < 0:
                continue
            # n-gram is from i - n + 1 to i.
            ngram = tuple(pred[(i - n + 1) : (i + 1)])
            # Update n-gram count.
            count_pred[ngram] += 1
            # Update p[n].
            p[n][1] += 1
            if count_pred[ngram] <= count_gold[ngram]:
                p[n][0] += 1
        scores.append(_compute_bleu(p, i + 1, len(gold), smooth))
    return scores


def score_corpus(preds, golds, ngrams, smooth=1e-5):
    golds = [ref for refs in golds for ref in refs]
    assert len(preds) == len(golds)
    p = []
    for n in range(ngrams + 1):
        p.append([0, 0])
    len_pred = len_gold = 0
    for pred, gold in zip(preds, golds):
        len_gold += len(gold)
        count_gold = defaultdict(int)
        _update_ngrams_count(gold, ngrams, count_gold)

        len_pred += len(pred)
        count_pred = defaultdict(int)
        _update_ngrams_count(pred, ngrams, count_pred)

        for k, v in count_pred.items():
            n = len(k)
            p[n][0] += min(v, count_gold[k])
            p[n][1] += v

    return _compute_bleu(p, len_pred, len_gold, smooth)


def _update_ngrams_count(sent, ngrams, count):
    length = len(sent)
    for n in range(1, ngrams + 1):
        for i in range(length - n + 1):
            ngram = tuple(sent[i : (i + n)])
            count[ngram] += 1


def _compute_bleu(p, len_pred, len_gold, smooth):
    # Brevity penalty.
    log_brevity = 1 - max(1, (len_gold + smooth) / (len_pred + smooth))
    log_score = 0
    ngrams = len(p) - 1
    for n in range(1, ngrams + 1):
        if p[n][1] > 0:
            if p[n][0] == 0:
                p[n][0] = 1e-16
            log_precision = math.log((p[n][0] + smooth) / (p[n][1] + smooth))
            log_score += log_precision
    log_score /= ngrams
    return math.exp(log_score + log_brevity)


if __name__ == '__main__':
    ids = []
    with open('./results/test_prediction_BeamSearch.1', 'r') as f:
        lines = f.readlines()
    comment_pred = []
    for i, line in enumerate(lines):
        comment_list = line.strip().split(' ')
        if comment_list[-1] == '<EOS>' and len(comment_list) > 2:
            comment_pred.append(comment_list[1:-1])
        else:
            comment_pred.append(comment_list[1:])
        # comment_pred.append(line.strip().split(' '))
        ids.append(i)

    with open('./dataset/java/test/javadoc.original', 'r') as f:
        lines = f.readlines()
    comment = []
    for line in lines:
        comment.append([line.strip().split(' ')])

    assert len(ids) == len(comment_pred) == len(comment)
    blue, rouge, meteor, _, _ = eval_bleu_rouge_meteor(ids, comment_pred, comment)
    print(blue, rouge, meteor)