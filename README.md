# adversarial-text

## Prepare Data

Download yelp_review_polarity_csv.tar.gz, amazon_review_polarity_csv.tar.gz to dataset via [http://goo.gl/JyCnZq](http://goo.gl/JyCnZq). Decompress the files.

Download trec07p.tgz to dataset via [https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html(https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html). Using the notebook dataset/trec_data.ipynb to preprocess the data.

## Train Model
``` bash
python word_model.py train --dataset amazon_review_polarity_csv --num_filters 300 --decay 2e-4 --tag conv300x1_dc2e4_l200 --gpu 1 --mem 0.5 
python word_model.py train --dataset trec07p --num_filters 512 --decay 2e-4 --tag lstm-mean --gpu 1 --mem 0.5
python deep_model.py train --dataset yelp_review_polarity_csv --blocks 1,1,1,1 -v 2-layer9 --gpu 1 --mem 0.5
```
