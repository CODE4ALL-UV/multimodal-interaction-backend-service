# multimodal-interaction-backend-service · Interacción multimodal

Este es el microservicio de **interacción multimodal** de Code4All: la parte
del backend que deja usar la app con algo distinto del teclado y el ratón. Hace
dos cosas con la cámara: **ver la mano del estudiante** para practicar el
alfabeto manual (dactilología) y **leer sus labios**. La segunda está explicada
en [Lectura de labios](#lectura-de-labios).

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

## Lectura de labios

Para estudiantes con discapacidad auditiva, y para cualquiera que quiera
practicar. La cámara dibuja los labios encima de la imagen, muestra cómo se
mueven y lee palabras sueltas de una lista. La app lo abre desde el menú de
perfil, solo en cuentas de estudiante.

Al contrario que en las señas, **aquí sí decide el servidor**: la forma de la
boca y la palabra se calculan en Python, en
`multimodal_service/domain/lip_reading.py`, y la app solo enseña el resultado.

### Qué se puede leer y qué no

La mayoría de las consonantes se hacen dentro de la boca: T, D, N, L, S, R, C, G
y J se ven casi igual. Por eso cada foto se reduce a una de siete formas: `M`
(labios juntos: M, B, P, V), `N` (entreabierta, sin forma propia) y las cinco
vocales. Las palabras se buscan **en una lista cerrada** (`VOCABULARY`), elegidas
porque se ven distintas. Una prueba impide añadir dos que se vean igual: «ayuda»,
«agua» y «pausa» no pueden estar juntas.

Es una ayuda para practicar, no un dictado ni una forma de evaluar a nadie.

### Por qué MediaPipe y no un modelo de lectura de labios

| Opción | Español | Tamaño | Licencia |
|---|---|---|---|
| **[MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)** (la elegida) | No aplica: da los puntos de la cara y aquí se calcula el resto | Modelo de 3,8 MB. La librería ya estaba instalada para las señas | Apache 2.0 |
| [Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) | Sí, con 44,5 % de palabras erradas | 186 MB más 180 MB del modelo de lenguaje, más PyTorch | Solo uso no comercial, para comparar o medir |
| [Auto-AVSR](https://github.com/mpc001/auto_avsr) | El sitio solo publica resultados con LRS3 | Modelos de 250 M de parámetros, más PyTorch | Código Apache 2.0; los modelos pueden tener otros términos |

Los modelos de lectura de labios leen frases libres, pero no caben en el plan
gratuito de Render, que es donde corre el gateway (512 MB de memoria), y con el
único modelo en español fallaría casi una de cada dos palabras. Si algún día hace
falta dictado libre, habría que desplegarlo aparte, con una máquina más grande y
revisando su licencia.

### Las rutas

#### `GET /api/lips/status`

Igual que el de las señas, más las palabras que sabe leer. Siempre contesta 200:

```json
{ "available": true, "model_ready": true, "reason": null, "words": ["hola", "..."] }
```

#### `POST /api/lips/landmarks`

Recibe una foto (`multipart/form-data`, campo `file`; JPEG, PNG o WebP, hasta
**1 MB**) y opcionalmente `neutral_width`. Es lo que la app dibuja en vivo.

```json
{
  "face": {
    "image": { "width": 720, "height": 480 },
    "lips": { "outer": [{ "x": 0.45, "y": 0.61 }], "inner": [{ "x": 0.46, "y": 0.61 }] },
    "opening": 0.26,
    "width": 0.6,
    "rounding": 0.02,
    "shape": "A",
    "scores": { "M": 0.0, "N": 0.0, "A": 0.99, "E": 0.01, "I": 0.0, "O": 0.0, "U": 0.0 }
  }
}
```

- `outer` e `inner` traen siempre 20 puntos cada uno, en orden, para dibujarlos
  como una línea. Van de 0 a 1 respecto a la foto.
- `image` es el tamaño de esa foto. Hace falta para dibujar los puntos: en un
  celular en vertical la vista previa recorta el vídeo apaisado, así que la foto
  y lo que se ve en pantalla no tienen la misma forma.
- `opening` y `width` van en distancias entre ojos: no cambian al acercarse.
- `rounding` va de 0 a 1 y sale de las blendshapes de MediaPipe.
- Si no se ve ninguna cara, `face` llega en `null`. **No es un error.**

#### `POST /api/lips/read`

Recibe las fotos de una palabra, en orden, repitiendo el campo `files` (hasta
**60 fotos y 16 MB**), y opcionalmente `neutral_width`.

```json
{
  "word": "hola",
  "alternatives": ["no"],
  "shapes": ["O", "A"],
  "frames": 24,
  "frames_with_face": 24,
  "reason": null
}
```

- `word` solo viene si la lectura es clara. Si no, llega en `null` y
  `alternatives` trae las candidatas, para que elija la persona.
- `reason` dice por qué no se intentó: `too_short`, `no_face` o `no_movement`.

| Código | Cuándo |
|---|---|
| 400 | Una foto llegó vacía, o en `/landmarks` no se pudo leer como imagen. En `/read`, una foto ilegible solo cuenta como una sin cara. |
| 413 | Una foto pasa de 1 MB, o el clip tiene demasiadas fotos o pesa demasiado. |
| 415 | Un archivo no es JPEG, PNG ni WebP. |
| 422 | No llegó el archivo, o `neutral_width` no está entre 0,2 y 1,2. |
| 503 | El servidor no puede detectar caras. El motivo va en el mensaje. |

### Cómo funciona por dentro

1. **Dos ritmos.** En vivo, la app manda una foto cuando el servidor termina con
   la anterior. Para leer una palabra toma unas 10 fotos por segundo, las guarda
   y las manda juntas al terminar: una vocal dura unas dos décimas de segundo, y
   el servidor gratuito no da abasto para verlas una a una.
2. **Se ajusta a cada cara.** La app mide el ancho de la boca relajada y lo manda
   en `neutral_width`. Una boca se ve redonda o estirada comparada con la suya.
3. **Cada foto da una puntuación por forma**, según la distancia a la forma típica
   (`_PROTOTYPES`). Salen de medidas publicadas de labios en vocales, no de vídeos
   de estudiantes: **si la lectura falla con personas reales, es lo primero que
   hay que ajustar**.
4. **La palabra se busca alineando.** Con programación dinámica se reparten las
   fotos entre las formas de cada palabra, en orden, para que no importe si se
   habla deprisa o despacio. Cuentan menos las fotos con la boca en pleno
   movimiento: al cerrar la boca desde una A se pasa por algo con forma de E, y eso
   no es decir una E.
5. **Solo se escribe si está clara**: la mejor palabra tiene que ganar a la segunda
   por `_MIN_MARGIN`. Si no, se ofrecen las candidatas.
6. **El análisis corre en un hilo aparte**, porque todos los servicios comparten
   proceso y, si no, el resto de rutas se quedaría esperando.

**Privacidad:** igual que en las señas, las fotos se procesan en memoria y se
descartan. El límite de 1 MB está para cumplirlo: por encima de eso el servidor
web las guardaría en un archivo temporal mientras llegan.

### Cuánto acierta

| Prueba | Resultado |
|---|---|
| Modelo real, foto real con la boca abierta a mano | Mide bien la apertura y pasa de `M` a `N`, `E` y `A`. Un clip de A larga deja «mal» entre las candidatas. |
| 29 palabras con hablantes simulados, a 10 fotos por segundo | Escribe bien el 63 %, escribe mal el 3 % y duda en el 34 %. La correcta queda a un toque el 99 % de las veces. |
| La misma simulación, a 6 fotos por segundo | Escribe bien el 49 %, escribe mal el 10 %. |
| **Personas reales hablando** | **Sin medir.** Es lo que falta. La simulación es optimista: usa las mismas formas que el lector. |

Mide la apertura y el ancho de los labios, así que `F` y los dientes no se
distinguen: `F` cuenta como una consonante oculta más.

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
├── domain/
│   └── lip_reading.py           medidas de la boca, formas y palabras
└── presentation/api/
    ├── sign_routes.py           las rutas de la mano, su modelo y su detector
    └── lip_routes.py            las rutas de los labios, su modelo y su detector
models/                          (no está en git) los modelos descargados
tests/
├── conftest.py
├── test_sign_routes.py
├── test_hand_detection.py
├── test_lip_reading.py
└── test_lip_routes.py
```

## Requisitos

- **Python:** `mediapipe`, `opencv-python-headless`, `numpy` y
  `python-multipart` (este último para recibir archivos en FastAPI). Están en
  `requirements.txt`.
- **Sistema:** MediaPipe necesita `libgl1`, `libegl1`, `libgles2` y
  `libglib2.0-0`. Sin libEGL el import funciona, pero falla al **crear** el
  detector. El `Dockerfile` del gateway ya las instala.
- **Internet** en la primera foto de cada función, para descargar su modelo.
  La lectura de labios no añade ninguna librería: usa las mismas que las señas.
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

Las de labios tampoco tocan la red: `test_lip_routes.py` sustituye MediaPipe por
un doble y `test_lip_reading.py` prueba la geometría y el lector con bocas
simuladas. Eso comprueba que el método funciona, no cuánto acierta con personas.

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
