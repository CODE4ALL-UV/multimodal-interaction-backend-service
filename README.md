# multimodal-interaction-backend-service
Repositorio Back-end para el módulo Interacción Multimodal.

Detección de la mano para reconocer el alfabeto manual con la cámara. Recibe
una foto y devuelve los 21 puntos de la mano que encuentra MediaPipe. **No
decide qué letra es**: eso lo hace la app, que así puede explicar qué dedo
falta estirar.

| Ruta | Para qué |
|---|---|
| `GET /api/signs/status` | Si el reconocimiento está disponible, sin usarlo. La app lo consulta antes de ofrecer la cámara. |
| `POST /api/signs/landmarks` | Los puntos de la mano en una foto. |

**Privacidad:** la imagen se procesa en memoria y se descarta. Nunca se guarda
ni se envía a otro servicio.

**Sistema:** MediaPipe necesita `libgl1`, `libegl1`, `libgles2` y
`libglib2.0-0`. El `Dockerfile` del gateway ya las instala. El modelo (7,8 MB)
se descarga solo en `models/` la primera vez que se usa.

## Cómo lo monta el gateway

`multimodal_service/routes.py` tiene `register(app)`, que añade las rutas de este
servicio a una aplicación de FastAPI. El gateway lo llama para cada servicio, y
`multimodal_service/main.py` hace lo mismo para arrancarlo solo.

## Correrlo

Lo normal es correrlo dentro del gateway (repo `Back-end`), que monta todos
los servicios juntos. Este no usa base de datos ni sesión, así que también
arranca solo, sin ningún otro repositorio:

```powershell
pip install -r requirements.txt
uvicorn multimodal_service.main:app --reload
```

## Pruebas

```bash
pytest tests
```
