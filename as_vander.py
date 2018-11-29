from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

dir="dataset.csv"
def dividir_data(dir,num):
    data = pd.read_csv(dir, error_bad_lines=False)
    data_n = data.drop(['ItemID', 'SentimentSource'], axis=1)
    return data_n.head(num)

dataset = dividir_data(dir,10)

analyzer = SentimentIntensityAnalyzer()
sentences=[]
for indice_fila, fila in dataset.iterrows():
    sentences.insert(indice_fila, fila['SentimentText'])

analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)).encode("utf-8"))
