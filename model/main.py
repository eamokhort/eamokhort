
import joblib
import numpy as np
import pandas as pd
import io
import inference
import cnn_ae

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

app=FastAPI()

app.mount("/static", StaticFiles(directory="public"))

filename = './model/model.pkl'
loaded_model = joblib.load(filename)

stsfilename = './model/stsc.pkl'
stsc = joblib.load(stsfilename)

class Model(BaseModel):
    X: list[str]

@app.get("/")

def read_root():
    return RedirectResponse('./static/index.html')

@app.post("/predict")
def predict_model(model: Model, ucl: float):
    df = pd.read_csv(io.StringIO(model.X), sep=';', index_col='datetime', parse_dates=True)
    result = inference.model_inference(df, loaded_model, stsc, ucl)
    return {"result" : ''.join(map(str, result))}

def main():
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000)

    if __name__ == "__main__":
        main()