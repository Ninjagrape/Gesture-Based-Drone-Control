import pickle as pkl 
import pandas as pd

with open("gesture_dynamic.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(r'gesture_dynamic.csv')
