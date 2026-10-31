"""Lectura de labios: de los puntos de la boca a formas y palabras.

Aquí no hay cámara, ni MediaPipe, ni red: solo geometría. Por eso se puede
probar entero sin descargar ningún modelo.

**Qué se puede leer en los labios y qué no.** La mayor parte de las consonantes
se hacen dentro de la boca: T, D, N, L, S, R, C, G y J se ven casi igual, y
leyendo solo los labios no se pueden separar. Lo que sí se ve son las vocales y
los labios que se juntan (M, B, P y V, que en español suena como B). Por eso:

- Cada foto se reduce a una de siete formas de boca.
- Las palabras se reconocen entre las de una lista cerrada, comparando el orden
  de las formas que se vieron con el que esa palabra debería producir. No es un
  dictado libre: eso exige modelos de cientos de megas que aquí no caben, y aun
  así fallan casi una de cada dos palabras en español.

Sirve para practicar y para escribir palabras sueltas. No reemplaza a nadie.
"""

import unicodedata
from dataclasses import dataclass

import numpy as np

# --- La boca en la malla de MediaPipe --------------------------------------

# Contornos de los labios, en orden, para poder dibujarlos como una línea.
LIPS_OUTER = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
)
LIPS_INNER = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
)

_UPPER_LIP = 13
_LOWER_LIP = 14
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_EYE_LEFT = 33
_EYE_RIGHT = 263

# Ancho de una boca relajada, en distancias entre ojos, para quien todavía no
# se ha calibrado. Varía de una persona a otra: por eso la app mide el suyo.
DEFAULT_NEUTRAL_WIDTH = 0.6


@dataclass(frozen=True)
class MouthMeasure:
    """Cómo está la boca en una foto.

    Las medidas van en distancias entre ojos, así que no cambian al acercarse
    o alejarse de la cámara.
    """

    opening: float
    width: float
    rounding: float = 0.0


def _point(landmark, image_width, image_height):
    # La malla da x e y relativas a cada lado de la foto y z relativa al ancho.
    # Sin pasar a píxeles, en una foto apaisada la boca parecería más estrecha.
    return np.array(
        [
            landmark.x * image_width,
            landmark.y * image_height,
            landmark.z * image_width,
        ]
    )


def measure_mouth(landmarks, image_width, image_height, rounding=0.0):
    """Mide la boca a partir de los 478 puntos de la cara."""

    def distance(first, second):
        return float(
            np.linalg.norm(
                _point(landmarks[first], image_width, image_height)
                - _point(landmarks[second], image_width, image_height)
            )
        )

    between_eyes = distance(_EYE_LEFT, _EYE_RIGHT)
    if between_eyes <= 0:
        return None

    return MouthMeasure(
        opening=distance(_UPPER_LIP, _LOWER_LIP) / between_eyes,
        width=distance(_MOUTH_LEFT, _MOUTH_RIGHT) / between_eyes,
        rounding=float(rounding),
    )


# --- Las formas de la boca -------------------------------------------------

# M: labios juntos (M, B, P, V). N: entreabierta, sin forma propia (T, D, N, L,
# S, R...). El resto son las vocales.
SHAPES = ("M", "N", "A", "E", "I", "O", "U")
_SHAPE_INDEX = {shape: index for index, shape in enumerate(SHAPES)}

# Cómo es cada forma: apertura, ancho respecto a la boca relajada y redondez.
#
# Salen de las medidas publicadas de labios en las vocales del español y el
# francés. No se ajustaron con vídeos de estudiantes, así que son el primer
# sitio donde mirar si la lectura falla con personas reales.
_PROTOTYPES = np.array(
    [
        [0.00, 1.00, 0.05],  # M
        [0.05, 1.00, 0.05],  # N
        [0.26, 1.00, 0.05],  # A
        [0.12, 1.06, 0.05],  # E
        [0.06, 1.14, 0.02],  # I
        [0.16, 0.76, 0.45],  # O
        [0.06, 0.66, 0.75],  # U
    ]
)

# Cuánto se puede alejar una boca de su forma sin dejar de serlo.
_SPREADS = np.array([0.035, 0.09, 0.25])


def shape_scores(mouth, neutral_width=DEFAULT_NEUTRAL_WIDTH):
    """Cuánto se parece la boca a cada forma. Las puntuaciones suman 1."""
    observed = np.array(
        [mouth.opening, mouth.width / neutral_width, mouth.rounding]
    )
    distances = (((_PROTOTYPES - observed) / _SPREADS) ** 2).sum(axis=1)
    # Se resta la menor para que una boca muy lejos de todas no se quede en
    # ceros por falta de precisión.
    likelihoods = np.exp(-0.5 * (distances - distances.min()))
    return likelihoods / likelihoods.sum()


def classify_shape(scores):
    return SHAPES[int(np.argmax(scores))]


# --- Las palabras ----------------------------------------------------------

# Elegidas para el curso y para pedir ayuda en clase, y porque en los labios se
# ven distintas. Muchas palabras útiles no pueden estar juntas: «ayuda», «agua»
# y «pausa» se ven igual (A-U-A), y lo mismo «error» y «perdón» (E-O). Una prueba
# comprueba que no se cuele ninguna pareja así.
VOCABULARY = (
    "hola", "gracias", "sí", "no", "ayuda", "por favor", "bien", "mal",
    "baño", "profesor", "clase", "entiendo", "repetir", "siguiente", "atrás",
    "empezar", "terminar", "python", "variable", "función", "bucle", "lista",
    "código", "imprimir", "número", "error", "ejercicio", "pregunta",
    "respuesta",
)

# Las que no se dicen como se escriben.
_PRONUNCIATION = {"python": "paiton"}

_VOWELS = frozenset("aeiou")


def _spoken_form(word):
    text = _PRONUNCIATION.get(word, word).lower()
    # La U de «que», «qui», «gue» y «gui» no suena. La de «pregunta» sí.
    text = text.replace("qu", "k").replace("gue", "ge").replace("gui", "gi")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def word_shapes(word):
    """Las formas que hace la boca al decir una palabra, en orden."""
    text = _spoken_form(word)
    shapes = []

    for index, letter in enumerate(text):
        following = text[index + 1] if index + 1 < len(text) else ""
        if letter == "h" or letter == " ":
            continue  # La H no suena y el espacio no se ve.
        if letter in _VOWELS:
            shape = letter.upper()
        elif letter == "y" and following not in _VOWELS:
            shape = "I"  # La Y final suena como una I.
        elif letter in "mbpv":
            shape = "M"
        else:
            shape = "N"

        if not shapes or shapes[-1] != shape:
            shapes.append(shape)

    return shapes


def _similarities(shape, *others):
    """Qué fotos cuentan como esta forma. 1 es esa forma y 0 otra muy distinta."""
    row = np.zeros(len(SHAPES))
    row[_SHAPE_INDEX[shape]] = 1.0
    for other, value in others:
        row[_SHAPE_INDEX[other]] = value
    return row


# Lo que cuesta confundir una forma con otra. O y U se parecen mucho; A y U,
# nada. Sin esto, una O dicha con la boca poco abierta contaría como un error
# tan grave como una A.
_SIMILAR = {
    "M": _similarities("M", ("N", 0.5)),
    "N": _similarities("N", ("M", 0.5), ("E", 0.5), ("I", 0.5)),
    "A": _similarities("A", ("E", 0.35), ("O", 0.3)),
    "E": _similarities("E", ("A", 0.35), ("I", 0.5), ("N", 0.5)),
    "I": _similarities("I", ("E", 0.5), ("N", 0.5)),
    "O": _similarities("O", ("U", 0.5), ("A", 0.3)),
    "U": _similarities("U", ("O", 0.5)),
}

# La boca en reposo, antes y después de la palabra.
_REST = _similarities("M", ("N", 1.0))

# Por qué no se pudo leer una palabra.
REASON_TOO_SHORT = "too_short"
REASON_NO_FACE = "no_face"
REASON_NO_MOVEMENT = "no_movement"

_MIN_FRAMES = 3

# Cuánta parte de las fotos tiene que tener cara para fiarse del resto.
_MIN_FACE_RATIO = 0.5

# Por debajo de esto la boca no se movió: unos 3 mm de abertura en una cara
# normal. Sin este filtro, una boca quieta acabaría «diciendo» alguna palabra.
_MIN_OPENING_RANGE = 0.035
_MIN_WIDTH_RANGE = 0.08
_MIN_ROUNDING_RANGE = 0.2

# Cuánto pesa no haber visto del todo una forma que la palabra necesita, frente
# al promedio foto a foto. Sin esto, «hola» se confundiría con «no»: una sola
# foto mal asignada apenas se nota en el promedio.
_MISSING_SHAPE_WEIGHT = 0.5

# Lo que vale una forma vista con la boca en pleno movimiento. Al cerrar la boca
# desde una A se pasa por algo con forma de E, pero nadie ha dicho una E: sin
# esto, una A larga se leía como «clase» (A-E).
_SHAPE_IN_MOTION_CREDIT = 0.3

# Cuánto pesa menos, en el promedio, una foto con la boca en pleno movimiento.
_FRAME_IN_MOTION_DISCOUNT = 0.5

# Por encima de este costo la palabra se parece poco, aunque sea la mejor.
_MAX_COST = 0.9

# Cuánto mejor que la segunda tiene que ser la primera para escribirla sola.
_MIN_MARGIN = 0.08

# Las palabras a esta distancia de la mejor se ofrecen como candidatas.
_ALTERNATIVE_MARGIN = 0.3
_MAX_ALTERNATIVES = 3


@dataclass(frozen=True)
class WordReading:
    """Lo que se leyó en un clip."""

    word: str | None
    alternatives: tuple
    shapes: tuple
    frames: int
    frames_with_face: int
    reason: str | None = None


def _states_of(shapes):
    """Reposo, las formas de la palabra y reposo.

    Las consonantes que no se ven (N) son opcionales: una T entre dos vocales
    no siempre deja ni una foto. Salvo entre dos formas iguales, porque ahí es
    lo único que separa A-T-A de una A larga.
    """
    rows = [_REST]
    required = [False]
    for index, shape in enumerate(shapes):
        rows.append(_SIMILAR[shape])
        separates = (
            0 < index < len(shapes) - 1 and shapes[index - 1] == shapes[index + 1]
        )
        required.append(shape != "N" or separates)
    rows.append(_REST)
    required.append(False)
    return np.array(rows), required


def _align(costs, required):
    """Reparte las fotos entre los estados, en orden, con el menor costo.

    Es programación dinámica: cada foto se queda en el estado de la anterior o
    avanza, y solo se puede saltar un estado si es opcional. Así no importa si
    alguien habla despacio o deprisa, y una palabra no puede encajar con las
    formas en otro orden.

    Devuelve el estado de cada foto, o None si no hay fotos para todas las
    formas que la palabra necesita.
    """
    frames, states = costs.shape
    if frames < sum(required):
        return None

    # De qué estados se puede venir a cada uno.
    sources = []
    for state in range(states):
        allowed = [state]
        previous = state - 1
        while previous >= 0:
            allowed.append(previous)
            if required[previous]:
                break
            previous -= 1
        sources.append(allowed)

    first_required = required.index(True) if True in required else states - 1
    last_required = (
        states - 1 - required[::-1].index(True) if True in required else 0
    )

    best = np.full((frames, states), np.inf)
    came_from = np.zeros((frames, states), dtype=int)
    best[0, : first_required + 1] = costs[0, : first_required + 1]

    for frame in range(1, frames):
        for state in range(states):
            for source in sources[state]:
                candidate = best[frame - 1, source] + costs[frame, state]
                if candidate < best[frame, state]:
                    best[frame, state] = candidate
                    came_from[frame, state] = source

    final = last_required + int(np.argmin(best[-1, last_required:]))
    if not np.isfinite(best[-1, final]):
        return None

    path = [final]
    for frame in range(frames - 1, 0, -1):
        path.append(int(came_from[frame, path[-1]]))
    path.reverse()
    return path


def _stillness(mouths, neutral_width):
    """Lo quieta que está la boca en cada foto, de 0 a 1, dentro de este clip."""
    speed = (
        np.abs(np.gradient([mouth.opening for mouth in mouths]))
        + 0.5 * np.abs(np.gradient([mouth.width / neutral_width for mouth in mouths]))
        + 0.3 * np.abs(np.gradient([mouth.rounding for mouth in mouths]))
    )
    return 1.0 - speed / max(float(speed.max()), 0.01)


def _word_cost(scores, shapes, stillness):
    """Cuánto le cuesta a esta palabra explicar el clip. Menos es mejor."""
    rows, required = _states_of(shapes)
    fit = scores @ rows.T
    costs = 1.0 - fit

    path = _align(costs, required)
    if path is None:
        return np.inf

    weights = 1.0 - _FRAME_IN_MOTION_DISCOUNT * (1.0 - stillness)
    # Se divide entre las fotos con la boca en movimiento y no entre todas: si
    # no, cuanto más reposo tuviera el clip, más se parecerían las palabras.
    active = max(1.0, float(((1.0 - scores @ _REST) * weights).sum()))
    frame_cost = float(
        sum(costs[t, s] * weights[t] for t, s in enumerate(path))
    ) / active

    # ¿Se llegó a ver bien cada forma que la palabra necesita, y con la boca
    # lo bastante quieta como para creérsela?
    credit = _SHAPE_IN_MOTION_CREDIT + (1.0 - _SHAPE_IN_MOTION_CREDIT) * stillness
    missing = [
        1.0 - max(fit[t, state] * credit[t] for t, s in enumerate(path) if s == state)
        for state, needed in enumerate(required)
        if needed and state in path
    ]
    # Una palabra con todas sus formas ocultas no tiene nada que comprobar.
    missing_cost = float(np.mean(missing)) if missing else 0.0

    return frame_cost + _MISSING_SHAPE_WEIGHT * missing_cost


def _moved(mouths, neutral_width):
    openings = [mouth.opening for mouth in mouths]
    widths = [mouth.width / neutral_width for mouth in mouths]
    roundings = [mouth.rounding for mouth in mouths]
    return (
        max(openings) - min(openings) >= _MIN_OPENING_RANGE
        or max(widths) - min(widths) >= _MIN_WIDTH_RANGE
        or max(roundings) - min(roundings) >= _MIN_ROUNDING_RANGE
    )


def _normalized_for_speaker(mouths):
    """Reduce la apertura de quien exagera, para que su E no parezca una A.

    Solo se reduce, nunca se aumenta: una boca poco abierta puede estar diciendo
    «sí», y no hay forma de distinguirlo de una A dicha sin abrir.
    """
    widest = float(np.percentile([mouth.opening for mouth in mouths], 95))
    scale = max(1.0, widest / _PROTOTYPES[_SHAPE_INDEX["A"], 0])
    return [
        MouthMeasure(mouth.opening / scale, mouth.width, mouth.rounding)
        for mouth in mouths
    ]


def _shapes_seen(labels):
    """Las formas del clip, sin repetir y sin las del reposo a los lados."""
    runs = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1][1] += 1
        else:
            runs.append([label, 1])

    # Con muchas fotos, una suelta suele ser la boca de paso entre dos formas.
    if len(labels) >= 10:
        runs = [run for run in runs if run[1] > 1]

    collapsed = []
    for label, _ in runs:
        if not collapsed or collapsed[-1] != label:
            collapsed.append(label)

    while collapsed and collapsed[0] in ("M", "N"):
        collapsed.pop(0)
    while collapsed and collapsed[-1] in ("M", "N"):
        collapsed.pop()
    return tuple(collapsed)


def read_word(mouths, neutral_width=DEFAULT_NEUTRAL_WIDTH, total_frames=None):
    """Busca qué palabra de la lista se dijo en un clip.

    [mouths] son las bocas de las fotos en las que se vio una cara, en orden.
    Si la boca no se parece lo bastante a ninguna palabra, o se parece a varias,
    no se elige: se devuelven las candidatas y que decida la persona. Adivinar
    aquí sería escribir por ella algo que no dijo.
    """
    total = len(mouths) if total_frames is None else total_frames

    def failure(reason):
        return WordReading(None, (), (), total, len(mouths), reason)

    if total < _MIN_FRAMES:
        return failure(REASON_TOO_SHORT)
    if len(mouths) < max(_MIN_FRAMES, total * _MIN_FACE_RATIO):
        return failure(REASON_NO_FACE)
    if not _moved(mouths, neutral_width):
        return failure(REASON_NO_MOVEMENT)

    normalized = _normalized_for_speaker(mouths)
    scores = np.array([shape_scores(mouth, neutral_width) for mouth in normalized])
    seen = _shapes_seen([classify_shape(row) for row in scores])
    stillness = _stillness(normalized, neutral_width)

    costs = {
        word: _word_cost(scores, word_shapes(word), stillness)
        for word in VOCABULARY
    }
    ranked = sorted(VOCABULARY, key=costs.get)
    best, runner_up = ranked[0], ranked[1]

    alternatives = tuple(
        word
        for word in ranked[1:]
        if costs[word] - costs[best] <= _ALTERNATIVE_MARGIN
    )[: _MAX_ALTERNATIVES]

    sure = (
        costs[best] <= _MAX_COST
        and costs[runner_up] - costs[best] >= _MIN_MARGIN
    )
    if not sure:
        # No se elige, pero la mejor encabeza las candidatas.
        return WordReading(
            None,
            (best, *alternatives)[: _MAX_ALTERNATIVES],
            seen,
            total,
            len(mouths),
        )

    return WordReading(best, alternatives, seen, total, len(mouths))
