# Rationalizing Neural Predictions Replication and Extension
## DS-GA 1011 Natural Language Processing Final Project

This repo replicates the Beer Advocate experiment as described in this [paper](https://people.csail.mit.edu/taolei/papers/emnlp16_rationale.pdf).

Acknowledgements for this code include the original paper's [source code](https://github.com/taolei87/rcnn/tree/master/code/rationale) and Adam Yala's PyTorch [implemention](https://github.com/yala/text_nn).

To run the model, first you need to obtain the Beer Advocate review data by going [here](http://people.csail.mit.edu/taolei/beer/).

1. Download the data, which are the files `reviews.aspect{#}.{dataset}.txt.gz`, to `"raw_data/beer_advocate/data"`.

2. Download the embeddings, which is the file `review+wiki.filtered.200.txt.gz`, to `"raw_data/beer_advocate/embeddings"`.

Then run the model using an example command like so:
```
python -m main --train --test --get_rationales --aspect appearance --epochs 1 --cuda --num_workers 1 --debug_mode
```

Use `--get_rationales` to enable extractive rationales.

The results and extracted rationales will be saved in `"results_path/args_dict.pkl"` and be accessed as

```
results = pickle.load(open('results_path/args_dict.pkl','rb'))
rationales = results['test_stats']['rationales']
```

`extracted_rationales.ipynb` provides an example on a model that was trained with `--debug_mode` and 10 epochs.
