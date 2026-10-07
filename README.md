# multimodal-interaction-backend-service · Interacción multimodal

Este es el microservicio de **interacción multimodal** de Code4All: la parte
del backend que deja usar la app con algo distinto del teclado y el ratón. Hoy
hace una sola cosa, que es **ver la mano del estudiante con la cámara** para
practicar el alfabeto manual (dactilología).

La app toma una foto con la cámara y la manda aquí. El servicio busca la mano
con [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
y devuelve los 21 puntos de la mano: la muñeca y las articulaciones y puntas de
cada dedo. **No decide qué letra es.** Eso lo hace la app con esos puntos, y
así puede decirle a la persona algo útil, como qué dedo le falta estirar, en
vez de un simple «incorrecto».

No guarda nada, no usa base de datos y no pide sesión.

## A quién le sirve

A estudiantes sordos o que se comunican en lengua de señas, y a cualquiera que
quiera practicar el alfabeto manual dentro del curso. La cámara se abre desde
las secciones de video del curso.

Una aclaración importante: es una **aproximación a la dactilología**, no un
traductor de Lengua de Señas Colombiana. La propia app lo advierte, y no
sustituye la validación de un intérprete.

## Las rutas

### `GET /api/signs/status`

Dice si el reconocimiento está disponible, sin usarlo. La app lo pregunta antes
de ofrecer la cámara, para no encenderla si el servidor no va a poder
responder. Siempre contesta 200:

```json
{ "available": true, "model_ready": true, "reason": null }
```

- `available`: si el servidor puede detectar manos.
- `model_ready`: si el modelo ya está descargado (ver más abajo).
- `reason`: por qué no está disponible, en castellano, cuando `available` es
  `false`. Por ejemplo, `"Faltan dependencias en el backend: ..."`.

Antes de la primera foto, `available` es optimista: dice `true` si las
librerías se pudieron importar, aunque el detector todavía no se haya creado.

### `POST /api/signs/landmarks`

Recibe una foto y devuelve los puntos de la mano.

- La foto va como **`multipart/form-data` en el campo `file`**. No acepta
  base64.
- Formatos: JPEG, PNG o WebP.
- Tamaño máximo: **4 MB**.

```json
{
  "hands": [
    {
      "handedness": "Right",
      "landmarks": [
        { "x": 0.52, "y": 0.81, "z": 0.0 },
        { "x": 0.47, "y": 0.74, "z": -0.02 }
      ]
    }
  ]
}
```

(En la respuesta real `landmarks` trae siempre los 21 puntos.)

- `x` e `y` van de 0 a 1, relativos al ancho y al alto de la foto. `z` es la
  profundidad relativa que calcula MediaPipe.
- `handedness` es la mano que cree ver MediaPipe (`Left` o `Right`).
- Se busca **una sola mano** por foto.
- Si no se ve ninguna mano, `hands` llega vacío. Eso **no es un error**: la app
  le pide a la persona que acerque o acomode la mano.

| Código | Cuándo |
|---|---|
| 400 | La foto llegó vacía o no se pudo leer como imagen. |
| 413 | La foto pesa más de 4 MB. |
| 415 | El archivo no es JPEG, PNG ni WebP. |
| 422 | No llegó el campo `file`. |
| 503 | El servidor no puede detectar manos: faltan librerías, no se pudo descargar el modelo o no se pudo crear el detector. El motivo va en el mensaje. |

## Cómo funciona por dentro

Todo está en `multimodal_service/presentation/api/sign_routes.py`.

1. **Las librerías se importan con cuidado.** Si `mediapipe`, `opencv` o
   `numpy` no están instaladas, el servicio (y el gateway con él) arranca
   igual. `/status` responde `available: false` y `/landmarks` responde 503.
   Así una instalación incompleta no tumba el resto del backend.
2. **El modelo se descarga solo la primera vez.** Es el modelo oficial de
   MediaPipe (`hand_landmarker.task`, float16, unos 7,8 MB). Se guarda en
   `models/` en la raíz del repo, que está en `.gitignore`. La descarga va a un
   archivo `.part` y después se renombra, para que una descarga cortada no deje
   un modelo a medias.
3. **Se crea un solo detector** y se reutiliza en todas las peticiones. Va en
   modo imagen (fotos sueltas, no video), con una mano como máximo y una
   confianza mínima de 0,4 para detectar la mano. Como el detector no se puede
   usar desde varios hilos a la vez, cada detección pasa por un candado.
4. **Con cada foto:** se valida el tipo y el tamaño, se decodifica con OpenCV,
   se pasa de BGR a RGB y se le entrega a MediaPipe. La respuesta es solo la
   lista de puntos.

**Privacidad:** la imagen se procesa en memoria y se descarta. Nunca se guarda
en disco ni se envía a otro servicio. Lo único que sale del servidor es la
descarga del modelo desde Google la primera vez.

## Qué hace la app con los puntos

Para que se entienda el recorrido completo: la clasificación vive en el
[Front-end](https://github.com/CODE4ALL-UV/Front-end), en
`lib/domain/models/sign_language/`.

- `hand_landmark_classifier.dart` mide qué tan estirado está cada dedo, dónde
  está el pulgar, si la mano apunta hacia abajo, si los dedos se cruzan y
  cuánto se abren, y lo compara con la forma esperada de cada letra en
  `hand_alphabet.dart`.
- Solo da una letra si la confianza llega a 0,5, y la da por segura desde 0,72
  si no hay otra letra parecida.
- **J, Z y Ñ no se pueden reconocer**, porque se hacen con movimiento y una
  foto fija no lo capta. Algunas letras con forma muy parecida (como H y U, o
  M, N, S y T) se confunden con facilidad, y en esos casos la app muestra las
  alternativas.
- `lib/data/services/sign_recognition_service.dart` es el que llama a estas
  dos rutas, con un tiempo de espera de 8 segundos.

## Cómo lo monta el gateway

Todos los servicios de Code4All siguen el mismo contrato:
`multimodal_service/routes.py` tiene una función `register(app)` que añade las
rutas de este servicio a una aplicación de FastAPI. El
[gateway](https://github.com/CODE4ALL-UV/Back-end) trae este repositorio como
submódulo en `services/` y llama a esa función al arrancar.
`multimodal_service/main.py` hace lo mismo para correrlo solo.

## Estructura

```
multimodal_service/
├── routes.py                    register(app): lo único que llama el gateway
├── main.py                      arranque independiente
└── presentation/api/
    └── sign_routes.py           las dos rutas, el modelo y el detector
models/                          (no está en git) el modelo descargado
tests/
├── conftest.py
└── test_sign_routes.py
```

## Requisitos

- **Python:** `mediapipe`, `opencv-python-headless`, `numpy` y
  `python-multipart` (este último para recibir archivos en FastAPI). Están en
  `requirements.txt`.
- **Sistema:** MediaPipe necesita `libgl1`, `libegl1`, `libgles2` y
  `libglib2.0-0`. Sin libEGL el import funciona, pero falla al **crear** el
  detector. El `Dockerfile` del gateway ya las instala.
- **Internet** en la primera foto, para descargar el modelo.
- **Variables de entorno:** ninguna.

## Correrlo

Lo normal es correrlo dentro del gateway (repo `Back-end`), que monta todos
los servicios juntos. Como este no usa base de datos ni sesión, también
arranca solo, sin ningún otro repositorio:

```powershell
pip install -r requirements.txt
uvicorn multimodal_service.main:app --reload
```

Para probarlo a mano, en `http://127.0.0.1:8000/docs` se puede subir una foto a
`/api/signs/landmarks`.

## Pruebas

```powershell
pytest tests
```

`tests/test_sign_routes.py` no toca la red ni necesita el modelo. Comprueba que
`/status` responde con sus tres campos, que un archivo que no es imagen recibe
415 y que una imagen vacía recibe 400.

En GitHub, cada push o pull request a `main` corre las pruebas con cobertura y
la sube a Codacy (`.github/workflows/codacy-coverage.yml`).

## Cosas a tener en cuenta

- **Es lo más pesado del backend.** MediaPipe y OpenCV ocupan mucho más que
  todo lo demás junto, y detectar una mano gasta CPU. Si algún día hay que
  sacar un servicio del gateway a su propio despliegue, este es el primer
  candidato.
- La detección corre dentro del proceso del gateway. Mientras se procesa una
  foto, el resto de rutas espera un momento.
- En Render el disco se borra con cada despliegue o reinicio, así que el
  modelo se vuelve a descargar con la primera foto después de cada uno. Esa
  primera foto tarda más.
- Si el detector no se puede crear (por ejemplo, por falta de libEGL), el
  servicio lo recuerda y no lo vuelve a intentar hasta que se reinicie.
- Las rutas son públicas, sin límite de peticiones.
- Las dependencias de `requirements.txt` no tienen versión fija. El código usa
  la API de tareas de MediaPipe (`mediapipe.tasks.python.vision`).

## Los repositorios de Code4All

| Parte | Repositorio |
|---|---|
| App (Flutter) | [Front-end](https://github.com/CODE4ALL-UV/Front-end) |
| API Gateway | [Back-end](https://github.com/CODE4ALL-UV/Back-end) |
| Gestión de usuarios | [user-management-backend-service](https://github.com/CODE4ALL-UV/user-management-backend-service) |
| Curso y contenidos de Python | [course-content-backend-service](https://github.com/CODE4ALL-UV/course-content-backend-service) |
| Ejercicios y evaluación | [assessment-backend-service](https://github.com/CODE4ALL-UV/assessment-backend-service) |
| Progreso y seguimiento | [progress-tracking-backend-service](https://github.com/CODE4ALL-UV/progress-tracking-backend-service) |
| Accesibilidad y adaptación | [accessibility-backend-service](https://github.com/CODE4ALL-UV/accessibility-backend-service) |
| **Interacción multimodal** | **este repositorio** |
| Infraestructura y dispositivos | [device-management-backend-service](https://github.com/CODE4ALL-UV/device-management-backend-service) |
| Capa de datos compartida (Neon) | [neon-storage-backend-service](https://github.com/CODE4ALL-UV/neon-storage-backend-service) |
