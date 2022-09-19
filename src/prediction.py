import random
import time
import json
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from dataloader import c2cDataset
from DECOM_model import DECOM
from utils import MaskedSoftmaxCELoss, DAMSMLoss, eval_bleu_rouge_meteor
import os

seed = 12345


def seed_everything(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def get_loaders(code_word2id, comment_word2id, dataset, max_code_len, max_comment_len, max_keywords_len,
                batch_size=32, num_workers=0, pin_memory=False):
    train_set = c2cDataset(code_word2id, comment_word2id, dataset, max_code_len, max_comment_len, max_keywords_len,
                           'train')
    test_set = c2cDataset(code_word2id, comment_word2id, dataset, max_code_len, max_comment_len, max_keywords_len,
                          'test')

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=train_set.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             collate_fn=test_set.collate_fn,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def evaluate_model(model, dataloader, bos_token, common_id2word, cuda, max_iter_num):
    losses, comment_reference, ids = [], [], []
    comment_prediction = {i: [] for i in range(max_iter_num + 1)}
    model.eval()

    seed_everything(seed)
    with torch.no_grad():
        for data in tqdm(dataloader):
            source_code, comment, template, keywords, source_code_len, comment_len, template_len, keywords_len = \
                [d.cuda() if cuda and not isinstance(d, list) else d for d in data[:8]]
            code_id = data[-1]

            bos = torch.tensor([bos_token] * len(comment), device=template.device).reshape(-1, 1)
            memory = model(source_code, bos, template, keywords,
                           source_code_len, comment_len, template_len, keywords_len)

            for i in range(len(comment)):
                ref = comment[i]
                comment_reference.append([ref])

                for j, comment_pred in enumerate(memory):
                    pre = [common_id2word[id] for id in comment_pred[i]]
                    comment_prediction[j].append(pre)

            ids += code_id

    for ii, comment_pred in enumerate(comment_prediction.values()):
        assert len(ids) == len(comment_pred) == len(comment_reference)
        bleu, rouge, meteor, _, _ = eval_bleu_rouge_meteor(ids, comment_pred, comment_reference)
        print(bleu, rouge, meteor)

    return bleu, rouge, meteor, comment_prediction


class Config(object):
    def __init__(self, dataset_config):
        dataset, max_code_len, max_comment_len, max_keywords_len = dataset_config.values()
        self.cuda = True
        self.dataset = dataset
        with open(fr'./../dataset/{dataset}/code.word2id', 'rb') as f:
            code_word2id = pickle.load(f)
        with open(fr'./../dataset/{dataset}/code.id2word', 'rb') as f:
            code_id2word = pickle.load(f)
        with open(fr'./../dataset/{dataset}/comment.word2id', 'rb') as f:
            comment_word2id = pickle.load(f)
        with open(fr'./../dataset/{dataset}/comment.id2word', 'rb') as f:
            comment_id2word = pickle.load(f)
        self.code_word2id = code_word2id
        self.code_id2word = code_id2word
        self.comment_word2id = comment_word2id
        self.comment_id2word = comment_id2word
        self.bos_token = self.comment_word2id['<BOS>']
        self.eos_token = self.comment_word2id['<EOS>']

        self.d_model = 512
        self.d_ff = 2048
        self.head_num = 8
        self.encoder_layer_num = 4
        self.decoder_layer_num = 6
        self.max_code_len = max_code_len
        self.max_comment_len = max_comment_len
        self.max_keywords_len = max_keywords_len
        self.code_vocab_size = len(code_word2id)
        self.comment_vocab_size = len(comment_word2id)
        self.beam_width = 5
        self.lr = 1e-4
        self.fineTune_lr = 1e-5
        self.batch_size = 32
        self.max_iter_num = 3
        self.dropout = 0.2
        self.epochs = 100
        self.clipping_distance = 16


if __name__ == '__main__':
    jcsd_config = {'name': 'JCSD', 'max_code_len': 300, 'max_comment_len': 50, 'max_keywords_len': 30}
    pcsd_config = {'name': 'PCSD', 'max_code_len': 100, 'max_comment_len': 50, 'max_keywords_len': 30}
    config = Config(jcsd_config)
    cuda = torch.cuda.is_available() and config.cuda
    if cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    seed_everything(seed)

    model = DECOM(config.d_model, config.d_ff, config.head_num, config.encoder_layer_num, config.decoder_layer_num,
                  config.code_vocab_size, config.comment_vocab_size, config.bos_token, config.eos_token,
                  config.max_comment_len, config.clipping_distance, config.max_iter_num, config.dropout,
                  config.beam_width)

    print("load the best model parameters!")
    model.load_state_dict(torch.load(f"./../saved_model/{config.dataset}/second_step_params.pkl"))

    # 模型装载至cuda
    if cuda:
        model.cuda()

    train_loader, test_loader = get_loaders(config.code_word2id, config.comment_word2id, config.dataset,
                                            config.max_code_len, config.max_comment_len, config.max_keywords_len,
                                            config.batch_size)

    test_Bleu, test_Rouge, test_Meteor, test_prediction = \
        evaluate_model(model, test_loader, config.bos_token, config.comment_id2word, cuda, config.max_iter_num)

    print('final_results: test_Bleu:{},test_Rouge:{},test_Meteor:{}'.
          format(test_Bleu, test_Rouge, test_Meteor))

    for ii, comment_pred in enumerate(test_prediction.values()):
        with open(f'./../results/{config.dataset}/beamSearch_result.{ii}', 'w') as w:
            for comment_list in comment_pred:
                comment = ' '.join(comment_list)
                w.write(comment + '\n')
