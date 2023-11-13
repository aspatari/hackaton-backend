from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import calculate_risk

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictIn(BaseModel):
    lon: float
    lat: float


@app.post("/predict")
def predict(predict_in: PredictIn):
    risk = calculate_risk(lon=predict_in.lon, lat=predict_in.lat)
    return risk


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
