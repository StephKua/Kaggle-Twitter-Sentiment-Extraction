# Kaggle-Twitter-Sentiment-Extraction - https://www.kaggle.com/c/tweet-sentiment-extraction/overview

# Ranking
- Top 2% - 30th Place

# Method
- Ensembling between 10 x Electra-Large and 10 x Roberta and XGB
- Preprocess and Postprocess on unknown tokens and grouped punctuations tokens for Roberta
- Error analysis on Predictions (backfired a little bit)
- XGB used to decide when to use original text for neutral tweets
- Used weighted confidence between all model's predictions to decide the final prediction
- Ensembling mostly done by (https://www.kaggle.com/css919)

## Models
1. Roberta
- pretrained with Squad2
- Multi-Sample Dropout (https://arxiv.org/pdf/1905.09788.pdf)
- Average Pooling of Last 4 Hidden Layer
- 5x Trained on all data but removed all tokens that are impossible to predict
- 5x Trained on Neutral Tweets only

2. Electra-Large
- fine tuned using TF scripts on colab with TPU and GCloud
- mostly done by my teammates (https://www.kaggle.com/ajinomoto132 and https://www.kaggle.com/tretrausaigon)

# Things tried but failed
- SWA
- ALBERT, ALBERT-LARGE
- LABEL SMOOTHING (probably didn't implement it correctly)
- Pretrain with More Tweets
- Exploit the original dataset
- Layer Wise LR Decay (probably didn't implement it correctly)
- Reproduce a customizable Electra on Pytorch
