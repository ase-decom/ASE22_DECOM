import pickle
from collections import Counter

from tqdm import tqdm


def build_vocab_pkl(code_vocab_size=50000, comment_vocab_size=50000, max_code_num=100, max_comment_len=50):
    file_list = ['train', 'test']
    code_token_dict, comment_token_dict = {}, {}
    for file in file_list:
        with open(fr'./dataset/rencos_python/{file}/source.code_original', 'r') as f:
            code_lines = f.readlines()

        with open(fr'./dataset/rencos_python/{file}/source.comment', 'r') as f:
            comment_lines = f.readlines()

        for code_line, comment_line in tqdm(zip(code_lines, comment_lines)):
            code_data = code_line.strip()
            for token in code_data.split(' ')[:max_code_num]:
                if not token.isspace():
                    if code_token_dict.get(token) is None:
                        code_token_dict[token] = 1
                    else:
                        code_token_dict[token] += 1

            comment_data = comment_line.strip()
            # comment_dict[file].append(comment_data)
            for token in comment_data.split(' ')[:max_comment_len]:
                if not token.isspace():
                    if comment_token_dict.get(token) is None:
                        comment_token_dict[token] = 1
                    else:
                        comment_token_dict[token] += 1

    print("num_code_token:", len(code_token_dict), "num_comment_token:", len(comment_token_dict))

    code_vocab = [tu[0] for tu in Counter(code_token_dict).most_common(code_vocab_size - 4)]
    comment_vocab = [tu[0] for tu in Counter(comment_token_dict).most_common(comment_vocab_size - 4)]
    code_vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + code_vocab
    comment_vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + comment_vocab
    print("num_code_vocab:", len(code_vocab), "num_comment_vocab:", len(comment_vocab))

    code_word2id = {word: idx for idx, word in enumerate(code_vocab)}
    code_id2word = {idx: word for idx, word in enumerate(code_vocab)}
    comment_word2id = {word: idx for idx, word in enumerate(comment_vocab)}
    comment_id2word = {idx: word for idx, word in enumerate(comment_vocab)}

    with open(fr'./dataset/rencos_python/code.word2id', 'wb') as w:
        pickle.dump(code_word2id, w)

    with open(fr'./dataset/rencos_python/code.id2word', 'wb') as w:
        pickle.dump(code_id2word, w)

    with open(fr'./dataset/rencos_python/comment.word2id', 'wb') as w:
        pickle.dump(comment_word2id, w)

    with open(fr'./dataset/rencos_python/comment.id2word', 'wb') as w:
        pickle.dump(comment_id2word, w)


if __name__ == '__main__':
    build_vocab_pkl()

    # from torch.utils.data import DataLoader
    #
    # dataset = c2cTrainDataset('test')
    # train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    # list(train_loader)
