import json
from collections import Counter

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle


class c2cDataset(Dataset):
    def __init__(self, code_word2id, comment_word2id, dataset, max_code_num, max_comment_len, max_keywords_len, file):
        self.ids = []
        self.source_code = []
        self.comment = []
        self.template = []
        self.keywords = []
        self.max_code_num = max_code_num
        self.max_comment_len = max_comment_len
        self.max_keywords_len = max_keywords_len
        self.code_word2id = code_word2id
        self.comment_word2id = comment_word2id
        self.file = file

        with open(fr'./../dataset/{dataset}/{file}/source.code', 'r', encoding="ISO-8859-1") as f:
            source_code_lines = f.readlines()
        with open(fr'./../dataset/{dataset}/{file}/source.comment', 'r', encoding="ISO-8859-1") as f:
            comment_lines = f.readlines()
        with open(fr'./../dataset/{dataset}/{file}/similar.comment', 'r', encoding="ISO-8859-1") as f:
            template_lines = f.readlines()
        with open(fr'./../dataset/{dataset}/{file}/source.keywords', 'r', encoding="ISO-8859-1") as f:
            identifier_lines = f.readlines()

        count_id = 0
        for code_line, comment_line, template_line, identifier_line in tqdm(
                zip(source_code_lines, comment_lines, template_lines, identifier_lines)):
            count_id += 1
            self.ids.append(count_id)

            code_token_list = code_line.strip().split(' ')
            source_code_list = [code_word2id[token] if token in code_word2id else code_word2id['<UNK>']
                                for token in code_token_list[:self.max_code_num]]
            self.source_code.append(source_code_list)

            if file != 'test':
                comment_token_list = comment_line.strip().split(' ')
                comment_list = [comment_word2id[token] if token in comment_word2id else comment_word2id['<UNK>']
                                for token in comment_token_list[:self.max_comment_len]] + [comment_word2id['<EOS>']]
                self.comment.append(comment_list)
            else:
                comment_token_list = comment_line.strip().split(' ')
                self.comment.append(comment_token_list)

            template_token_list = template_line.strip().split(' ')
            template_list = [comment_word2id[token] if token in comment_word2id else comment_word2id['<UNK>']
                             for token in template_token_list[:self.max_comment_len]]
            self.template.append(template_list)

            identifier_token_list = identifier_line.strip().split(' ')
            identifier_list = [comment_word2id[token] for token in identifier_token_list
                               if token in comment_word2id]
            self.keywords.append(identifier_list[:self.max_keywords_len])

    def __getitem__(self, index):
        return self.source_code[index], \
               self.comment[index], \
               self.template[index], \
               self.keywords[index], \
               len(self.source_code[index]), \
               len(self.comment[index]), \
               len(self.template[index]), \
               len(self.keywords[index]), \
               self.ids[index]

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return_list = []
        for i in dat:
            if i < 4:
                if i == 1 and self.file == 'test':
                    return_list.append(dat[i].tolist())
                else:
                    return_list.append(pad_sequence([torch.tensor(x, dtype=torch.int64) for x in dat[i].tolist()], True))
            elif i < 8:
                if i == 5 and self.file == 'test':
                    return_list.append(dat[i].tolist())
                else:
                    return_list.append(torch.tensor(dat[i].tolist()))
            else:
                return_list.append(dat[i].tolist())
        return return_list


# class c2cEvalDataset(Dataset):
#     def __init__(self, file, code_word2id, comment_word2id, dataset='rencos_java',
#                  max_code_num=100, max_comment_len=50, max_keywords_len=30):
#         self.ids = []
#         self.source_code = []
#         self.comment = []
#         self.template = []
#         self.keywords = []
#         self.max_code_num = max_code_num
#         self.max_comment_len = max_comment_len
#         self.max_keywords_len = max_keywords_len
#         self.code_word2id = code_word2id
#         self.comment_word2id = comment_word2id
#
#         with open(fr'./dataset/{dataset}/{file}/source.code_original', 'r') as f:
#             source_code_lines = f.readlines()
#         with open(fr'./dataset/{dataset}/{file}/source.comment', 'r') as f:
#             comment_lines = f.readlines()
#         with open(fr'./dataset/{dataset}/{file}/similar.comment', 'r') as f:
#             template_lines = f.readlines()
#         with open(fr'./dataset/{dataset}/{file}/source.identifier', 'r') as f:
#             identifier_lines = f.readlines()
#
#         count_id = 0
#         for code_line, comment_line, template_line, identifier_line in tqdm(
#                 zip(source_code_lines, comment_lines, template_lines, identifier_lines)):
#             count_id += 1
#             self.ids.append(count_id)
#
#             code_token_list = code_line.strip().split(' ')
#             source_code_list = [code_word2id[token] if token in code_word2id else code_word2id['<UNK>']
#                                 for token in code_token_list[:self.max_code_num]]
#             self.source_code.append(source_code_list)
#             # self.source_code.append(source_code_list)
#
#             comment_token_list = comment_line.strip().split(' ')
#             self.comment.append(comment_token_list)
#
#             template_token_list = template_line.strip().split(' ')
#             template_list = [comment_word2id[token] if token in comment_word2id else comment_word2id['<UNK>']
#                              for token in template_token_list[:self.max_comment_len]]
#             self.template.append(template_list)
#             # self.template.append([common_word2id['<PAD>']])
#
#             identifier_token_list = identifier_line.strip().split(' ')
#             identifier_list = [comment_word2id[token] for token in identifier_token_list
#                                if token in comment_word2id]
#             if len(identifier_list) == 0:
#                 print(count_id, identifier_line)
#             self.keywords.append(identifier_list[:self.max_keywords_len])
#
#     def __getitem__(self, index):
#         return self.source_code[index], \
#                self.template[index], \
#                self.keywords[index], \
#                len(self.source_code[index]), \
#                len(self.template[index]), \
#                len(self.keywords[index]), \
#                self.comment[index], \
#                len(self.comment[index]), \
#                self.ids[index]
#
#     def __len__(self):
#         return len(self.ids)
#
#     # dataloader自定义padding操作
#     def collate_fn(self, data):
#         dat = pd.DataFrame(data)
#         return_list = []
#         for i in dat:
#             if i < 3:
#                 return_list.append(pad_sequence([torch.tensor(x) for x in dat[i].tolist()], True))
#             elif i < 6:
#                 return_list.append(torch.tensor(dat[i].tolist()))
#             else:
#                 return_list.append(dat[i].tolist())
#         return return_list
