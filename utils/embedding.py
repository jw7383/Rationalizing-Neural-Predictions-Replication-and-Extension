import gzip
import numpy as np
import torch
import tqdm


def get_embedding_tensor(args):
    if args.embedding == 'beer':
        embedding_path='raw_data/beer_advocate/embeddings/review+wiki.filtered.200.txt.gz'
        lines = []
        with gzip.open(embedding_path, mode = 'rt') as file:
            lines = file.readlines()
            file.close()
        embedding_tensor = []
        token2id = {}
        token2id['<unk>'] = 0
        token2id['<pad>'] = 1
        for id, l in enumerate(lines):
            word, emb = l.split()[0], l.split()[1:]
            vector = [float(x) for x in emb ]
            if id == 0:
                embedding_tensor.append( np.zeros( len(vector) ) )
                embedding_tensor.append( np.zeros( len(vector) ) )
            embedding_tensor.append(vector)
            token2id[word] = id + 2
        embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
        args.embedding_dim = embedding_tensor.shape[1]
    elif args.embedding == 'glove':
        embedding_path='raw_data/rotten_tomatoes/embeddings/glove.6B.200d.txt'
        lines = []
        with open(embedding_path, encoding='utf8') as file:
            lines = file.readlines()
            file.close()
        embedding_tensor = []
        token2id = {}
        token2id['<unk>'] = 0
        token2id['<pad>'] = 1
        for id, l in tqdm.tqdm(enumerate(lines)):
            word, emb = l.split()[0], l.split()[1:]
            vector = [float(x) for x in emb ]
            if id == 0:
                embedding_tensor.append( np.zeros( len(vector) ) )
                embedding_tensor.append( np.zeros( len(vector) ) )
            embedding_tensor.append(vector)
            token2id[word] = id + 2
        embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    args.embedding_dim = embedding_tensor.shape[1]
    return embedding_tensor, token2id

def get_indices_tensor(text_list, token2id, max_length):
    '''
    -text_list: array of word tokens
    -token2id: mapping of word -> index
    -max length of return tokens
    returns tensor of same size as text with each word's corresponding index
    '''
    UNK_ID = 0 
    PAD_ID = 1  
    text_id = [ token2id[word] if word in token2id else UNK_ID for word in text_list][:max_length]
    if len(text_id) < max_length:
        text_id.extend( [ PAD_ID for _ in range(max_length - len(text_id))])

    x =  torch.LongTensor([text_id])

    return x
