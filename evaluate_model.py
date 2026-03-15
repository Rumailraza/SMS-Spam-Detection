import pickle
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/spam.csv")

model = pickle.load(open("models/spam_model.pkl","rb"))