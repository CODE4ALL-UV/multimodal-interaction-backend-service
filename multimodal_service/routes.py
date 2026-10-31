"""Lo que este microservicio aporta al API Gateway.

El gateway (repo Back-end) llama a `register(app)` de cada servicio, y el
`main.py` de este paquete hace lo mismo para arrancarlo solo. Así las rutas se
declaran una sola vez, se arranque como se arranque.
"""

from fastapi import FastAPI

from multimodal_service.presentation.api.lip_routes import router as lip_router
from multimodal_service.presentation.api.sign_routes import router as sign_router


def register(app: FastAPI) -> None:
    app.include_router(sign_router)
    app.include_router(lip_router)
