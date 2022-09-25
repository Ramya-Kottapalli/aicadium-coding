import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# $1 = test dataset path, $2= model path, $3= feature selection bool (0 or 1)
app = Flask(__name__)

features = ['PageValues', 'Month', 'VisitorType', 'BounceRates', 'ProductRelated_Duration', 
'ProductRelated', 'Administrative_Duration', 'Informational', 'Administrative', 'ExitRates', 'SpecialDay']

select = sys.argv[3]
model = XGBClassifier()
model.load_model(sys.argv[2])

# same preprocssing applied to training data - categorical to numerical and feature selection
def preprocess(df):
    label_encode = ['Month', 'VisitorType']
    le = LabelEncoder()

    df[label_encode] = df[label_encode].apply(lambda col: le.fit_transform(col))

    if select==1:
        #print("selecting")
        df = df[features]

    return df

@app.route('/')
def predict():
    df = pd.read_csv(sys.argv[1])
    df.drop(df.columns[[0]], axis=1, inplace=True)

    df = preprocess(df)
    y_pred = model.predict(df)

    df["PredictedRevenue"] = y_pred
    
    html = (
    df.style
    .render())
    return html

app.run(port=5000, debug=True, use_reloader=False)