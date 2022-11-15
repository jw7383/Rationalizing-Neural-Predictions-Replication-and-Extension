import gzip
import tqdm
import json
from utils.embedding import get_indices_tensor

def get_dataset(args, token2id):
    train = process_file(args, token2id, 'train')
    dev = process_file(args, token2id, 'dev')
    test = process_file(args, token2id, 'test')
    
    return train, dev, test

def get_rationale(args, token2id):
    rationale = process_file(args, token2id, 'rationale')
    return rationale

def process_file(args, token2id, dataset):
    if dataset != 'rationale':
        if args.dataset == 'beer_advocate':
            root_path = 'raw_data/beer_advocate/data/reviews.aspect'
        data = []
        token2id  = token2id
        aspects_id_map = {'appearance':0, 'aroma':1, 'palate':2,'taste':3}
        name_key_map = {'train':'train', 'dev':'heldout', 'test':'heldout'}
        with gzip.open(root_path+str(aspects_id_map[args.aspect])+'.'+name_key_map[dataset]+'.txt.gz') as gfile:
            lines = gfile.readlines()
            lines = list(zip( range(len(lines)), lines) )
            if args.debug_mode:
                lines = lines[:800]
            elif dataset == 'dev':
                lines = lines[:5000]
            elif dataset == 'test':
                lines = lines[5000:10000]
            elif dataset == 'train':
                lines = lines[0:20000]

            for id, line in tqdm.tqdm(enumerate(lines)):
                uid, line_content = line
                sample = process_line(args, token2id, line_content, aspects_id_map[args.aspect], id)
                sample['uid'] = uid
                data.append(sample)
            gfile.close()
    elif dataset == 'rationale':
        json_lines = [json.loads(line) for line in open('raw_data/beer_advocate/annotations/annotations.json', 'r', encoding='utf-8')]
        data = []
        for id, json_line in enumerate(json_lines):
            text = " ".join(json_line['x'])
            x = get_indices_tensor(json_line['x'], token2id, max_length = 250)
            if args.objective == 'mse':
                label = float(json_line['raw'].split(f"""'review/{args.aspect}': """)[1][0:3])/5.0
                args.num_class = 1
            else:
                class_map = {0: 0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1, 8:2, 9:2, 10:2}
                float_label = float(json_line['raw'].split(f"""'review/{args.aspect}': """)[1][0:3])/5.0
                label = int(class_map[int(float_label*10)])
                args.num_class = 3
            sample = {'text':text,'x':x, 'y':label, 'i':id}
            data.append(sample)
    return data

## Convert one line from beer dataset to {Text, Tensor, Labels}
def process_line(args, token2id, line, aspect_id, id):
    max_length = 250
    class_map = {0: 0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1, 8:2, 9:2, 10:2} #for cross_entropy only
    if isinstance(line, bytes):
        line = line.decode()
    labels = [ float(v) for v in line.split()[:5] ]
    if args.objective== 'mse':
        label = float(labels[aspect_id])
        args.num_class = 1
    else:
        label = int(class_map[ int(labels[aspect_id] *10) ])
        args.num_class = 3
    text_list = line.split('\t')[-1].split()[:max_length]
    text = " ".join(text_list)
    x =  get_indices_tensor(text_list, token2id, max_length)
    sample = {'text':text,'x':x, 'y':label, 'i':id}
    return sample
