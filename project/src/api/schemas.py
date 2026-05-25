from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class PredictionResponse(BaseModel):
    image_name: str
    bbox: BoundingBox
    confidence: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str