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

### Adding your own data set 
Create a folder with the name of dataset in dataset/, and store data in train.json and test.json with the following structure.
```json
{
  "texts" : [
    "Text Sample 1",
    "Text Sample 2"
  ],
  "labels" : [
    "label1",
    "label2"
  ]
}
```
## Train Lanugage Model

Run notebook build_lm.ipynb. You need to specify `KENLM_PATH` to the path of executable files of kenlm.

## Train Model
Using word_mode.py for LSTM and word level CNN, deep_model.py for deep character-level CNN. Some argumentments
+ first argument is action, for training model it should be "train".
+ --dataset the name of dataset
+ --num_filters number of in hidden units, used by LSTM and word level CNN.
+ --decay weight decay
+ --tag specify the type of model, could be lstm* or conv*. The prefix is used to determine the structure of model. The whole tag is used to create a folder to save the weights of model.
+ -v specify the type of model, use with deep_model.py. Now we use type 2 for most experiments. So it should be 2*. The whole tag is used to create a folder to save the weights of model.
+ --blocks number of convolution blocks in each section, 1 convolution blocks consist of 2 conv. So `1,1,1,1` means 8 convolutions in total.

### Examples:
``` bash
python word_model.py train --dataset yelp_review_polarity_csv --num_filters 512 --decay 2e-4 --tag lstm --gpu 1 --mem 0.5
python word_model.py train --dataset yelp_review_polarity_csv --num_filters 300 --decay 2e-4 --tag conv --gpu 1 --mem 0.5
python deep_model.py train --dataset yelp_review_polarity_csv --blocks 1,1,1,1 -v 2-layer9 --gpu 1 --mem 0.5
```

## Load Model
For loading the model, we need to train the action from `train` to `notebook`. To execute the python script inside a notebook we use cell magic `%run`. After loading the model, the model can be accessed by name `model`, we keep the name in a dictionary and then load other models. Like
``` ipython
models = dict()
%run word_model.py notebook --dataset yelp_review_polarity_csv --num_filters 512 --decay 2e-4 --tag lstm --gpu 1 --mem 0.5
models['LSTM'] = model
%run word_model.py notebook --dataset yelp_review_polarity_csv --num_filters 300 --decay 2e-4 --tag conv --gpu 1 --mem 0.5
models['WordCNN'] = model
%run deep_model.py notebook --dataset yelp_review_polarity_csv --blocks 1,1,1,1 -v 2-layer9 --gpu 1 --mem 0.5
models['VDCNN-11'] = model
```
