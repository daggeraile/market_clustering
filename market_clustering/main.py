import pandas as pd
import numpy as np

url = "https://www.kaggle.com/lancengck/marketing-data/download"
data = pd.read_csv(url)

print(data[0])