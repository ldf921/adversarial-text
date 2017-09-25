# adversarial-text

## Train Word Level Model
``` bash
python word_model.py train --dataset amazon_review_polarity_csv --num_filters 300 --decay 2e-4 --tag conv300x1_dc2e4_l200 --gpu 1 --mem 0.5 
python word_model.py train --dataset amazon_review_polarity_csv --num_filters 512 --decay 2e-4 --tag lstm-mean --gpu 1 --mem 0.5
```
