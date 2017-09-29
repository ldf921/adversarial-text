# adversarial-text

## Prerequisites
```sh
pip install gensim
pip install h5py
pip install nltk
```
Install kenlm from the [repo](https://github.com/kpu/kenlm).


Download word2vec word embeddings [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), extract file to GoogleNews-vectors-negative300.bin.

Download counter-fitting word vectors by running download-counter-fitting.sh.

## Prepare Data

Download yelp_review_polarity_csv.tar.gz, amazon_review_polarity_csv.tar.gz to dataset via [http://goo.gl/JyCnZq](http://goo.gl/JyCnZq). Decompress the files.

Download trec07p.tgz to dataset via [https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html](https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html). Using the notebook dataset/trec_data.ipynb to preprocess the data.

## Train Lanugage Model

Run notebook build_lm.ipynb. You need to specify `KENLM_PATH` to the path of executable files of kenlm.

## Train Model
``` bash
python word_model.py train --dataset yelp_review_polarity_csv --num_filters 512 --decay 2e-4 --tag lstm --gpu 1 --mem 0.5
python word_model.py train --dataset yelp_review_polarity_csv --num_filters 300 --decay 2e-4 --tag conv --gpu 1 --mem 0.5
python deep_model.py train --dataset yelp_review_polarity_csv --blocks 1,1,1,1 -v 2-layer9
```

## Adversarial Generation
See run_training.ipynb
