# Rationalizing Neural Predictions Replication and Extension
## DS-GA 1011 Natural Language Processing Final Project

This repo replicates the Beer Advocate experiment as described in this [paper](https://people.csail.mit.edu/taolei/papers/emnlp16_rationale.pdf).

To run the model, first you need to obtain the Beer Advocate review data by going [here](http://people.csail.mit.edu/taolei/beer/).

1. Download the data, which are the files reviews.aspect{#}.{dataset}.txt.gz, to raw_data/beer_advocate/data.

2. Download the embeddings, which is the file review+wiki.filtered.200.txt.gz, to raw_data/beer_advocate/embeddings.

Then run the model using an example command like so:
```
python -m main --train --test --get_rationales --epochs 10 --cuda --num_workers 1
```

