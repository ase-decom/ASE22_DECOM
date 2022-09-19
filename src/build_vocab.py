import pickle
from collections import Counter

from tqdm import tqdm


def build_vocab_pkl(config):
    name, code_vocab_size, comment_vocab_size, max_code_len, max_comment_len = config.values()
    file_list = ['train', 'valid', 'test']
    code_token_dict, comment_token_dict = {}, {}
    for file in file_list:
        with open(fr'./../dataset/{name}/{file}/source.code', 'r', encoding="ISO-8859-1") as f:
            code_lines = f.readlines()

        with open(fr'./../dataset/{name}/{file}/source.comment', 'r', encoding="ISO-8859-1") as f:
            comment_lines = f.readlines()

        for code_line, comment_line in tqdm(zip(code_lines, comment_lines)):
            code_data = code_line.strip()
            for token in code_data.split(' ')[:max_code_len]:
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

    with open(fr'./../dataset/{name}/code.word2id', 'wb') as w:
        pickle.dump(code_word2id, w)

    with open(fr'./../dataset/{name}/code.id2word', 'wb') as w:
        pickle.dump(code_id2word, w)

    with open(fr'./../dataset/{name}/comment.word2id', 'wb') as w:
        pickle.dump(comment_word2id, w)

    with open(fr'./../dataset/{name}/comment.id2word', 'wb') as w:
        pickle.dump(comment_id2word, w)


if __name__ == '__main__':
    jcsd_config = {'name': 'JCSD', 'code_vocab_size': 50000, 'comment_vocab_size': 50000,
                   'max_code_len': 300, 'max_comment_len': 50}
    pcsd_config = {'name': 'PCSD', 'code_vocab_size': 50000, 'comment_vocab_size': 50000,
                   'max_code_len': 100, 'max_comment_len': 50}
    build_vocab_pkl(pcsd_config)
