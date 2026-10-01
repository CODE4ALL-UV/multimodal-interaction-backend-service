"""Arranca solo este microservicio, sin el gateway.

Sirve para desarrollarlo o desplegarlo por separado. En producción lo monta el
API Gateway (repo Back-end) junto a los demás. No usa base de datos ni sesión,
así que no depende de ningún otro repositorio:

    uvicorn multimodal_service.main:app --reload

Es el candidato natural a desplegarse aparte: MediaPipe y OpenCV son, con
diferencia, lo más pesado del backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from multimodal_service.routes import register

app = FastAPI(title="Multimodal Interaction Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register(app)
