"""Las medidas de la boca, sus formas y la lectura de palabras.

No hace falta cámara ni modelo: aquí solo hay geometría. Las palabras se prueban
con bocas simuladas a partir de las mismas formas que usa el lector. Eso
comprueba que el método funciona, no cuánto acierta con personas de verdad.
"""

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from multimodal_service.domain import lip_reading
from multimodal_service.domain.lip_reading import (
    DEFAULT_NEUTRAL_WIDTH,
    LIPS_INNER,
    LIPS_OUTER,
    REASON_NO_FACE,
    REASON_NO_MOVEMENT,
    REASON_TOO_SHORT,
    SHAPES,
    VOCABULARY,
    MouthMeasure,
    classify_shape,
    measure_mouth,
    read_word,
    shape_scores,
    word_shapes,
)


def _shape(shape):
    opening, ratio, rounding = lip_reading._PROTOTYPES[SHAPES.index(shape)]
    return MouthMeasure(opening, ratio * DEFAULT_NEUTRAL_WIDTH, rounding)


def _face(opening=0.0, width=0.6, between_eyes=100.0):
    """Los puntos de una cara, en píxeles de una foto cuadrada de 1000."""
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    def place(index, x, y):
        points[index] = SimpleNamespace(x=x / 1000, y=y / 1000, z=0.0)

    place(33, 500 - between_eyes / 2, 400)
    place(263, 500 + between_eyes / 2, 400)
    place(61, 500 - between_eyes * width / 2, 600)
    place(291, 500 + between_eyes * width / 2, 600)
    place(13, 500, 600 - between_eyes * opening / 2)
    place(14, 500, 600 + between_eyes * opening / 2)
    return points


# --- Medir la boca ---------------------------------------------------------


def test_measures_opening_and_width_in_distances_between_eyes():
    mouth = measure_mouth(_face(opening=0.2, width=0.6), 1000, 1000, rounding=0.3)

    assert mouth.opening == pytest.approx(0.2)
    assert mouth.width == pytest.approx(0.6)
    assert mouth.rounding == pytest.approx(0.3)


def test_measures_do_not_change_when_the_face_gets_closer():
    near = measure_mouth(_face(0.2, 0.6, between_eyes=200), 1000, 1000)
    far = measure_mouth(_face(0.2, 0.6, between_eyes=80), 1000, 1000)

    assert near.opening == pytest.approx(far.opening)
    assert near.width == pytest.approx(far.width)


def test_measures_use_pixels_not_the_relative_coordinates():
    # La misma cara en una foto apaisada: x e y relativas ya no miden lo mismo.
    points = _face(opening=0.2, width=0.6)
    for point in points:
        point.x = point.x * 1000 / 2000

    mouth = measure_mouth(points, 2000, 1000)

    assert mouth.opening == pytest.approx(0.2)
    assert mouth.width == pytest.approx(0.6)


def test_a_face_without_size_has_no_measure():
    assert measure_mouth(_face(between_eyes=0.0), 1000, 1000) is None


def test_lip_contours_are_the_ones_mediapipe_declares():
    assert len(LIPS_OUTER) == 20
    assert len(LIPS_INNER) == 20
    assert not set(LIPS_OUTER) & set(LIPS_INNER)


# --- Las formas de la boca -------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_each_shape_is_recognized_from_its_own_mouth(shape):
    scores = shape_scores(_shape(shape))

    assert classify_shape(scores) == shape
    assert scores.sum() == pytest.approx(1.0)


def test_a_mouth_far_from_every_shape_still_gives_valid_scores():
    scores = shape_scores(MouthMeasure(opening=3.0, width=9.0, rounding=1.0))

    assert np.isfinite(scores).all()
    assert scores.sum() == pytest.approx(1.0)


def test_rounding_is_judged_against_the_persons_own_relaxed_mouth():
    # Una boca de 0,5 es estrecha en una cara ancha y normal en una estrecha.
    mouth = MouthMeasure(opening=0.06, width=0.5, rounding=0.3)

    wide_face = classify_shape(shape_scores(mouth, neutral_width=0.8))
    narrow_face = classify_shape(shape_scores(mouth, neutral_width=0.5))

    assert wide_face == "U"
    assert narrow_face != "U"


# --- Cómo se ve cada palabra -----------------------------------------------


@pytest.mark.parametrize(
    ("word", "shapes"),
    [
        ("hola", ["O", "N", "A"]),  # La H no suena.
        ("baño", ["M", "A", "N", "O"]),
        ("variable", ["M", "A", "N", "I", "A", "M", "N", "E"]),  # V como B.
        ("por favor", ["M", "O", "N", "A", "M", "O", "N"]),  # El espacio no se ve.
        ("sí", ["N", "I"]),  # Sin tilde.
        ("python", ["M", "A", "I", "N", "O", "N"]),  # Se dice «paiton».
        ("siguiente", ["N", "I", "N", "I", "E", "N", "E"]),  # La U de «gui» no.
        ("pregunta", ["M", "N", "E", "N", "U", "N", "A"]),  # La de «gun» sí.
        ("ey", ["E", "I"]),  # La Y final suena como una I.
    ],
)
def test_words_become_the_shapes_they_make_on_the_lips(word, shapes):
    assert word_shapes(word) == shapes


def test_no_two_words_of_the_list_look_the_same_on_the_lips():
    looks = defaultdict(list)
    for word in VOCABULARY:
        looks[tuple(word_shapes(word))].append(word)

    assert [words for words in looks.values() if len(words) > 1] == []


# --- Leer palabras ---------------------------------------------------------


def _say(word, rng, fps=10.0, speaker=True, tempo=1.0):
    """Una boca diciendo una palabra: reposo, cada forma y reposo otra vez.

    `speaker` varía cuánto abre y redondea la boca cada persona; `tempo` alarga
    cada forma, como quien vocaliza despacio.
    """
    open_scale = rng.uniform(0.75, 1.3) if speaker else 1.0
    round_scale = rng.uniform(0.5, 1.1) if speaker else 1.0

    keys = [("M", 0.25)]
    for shape in word_shapes(word):
        keys.append((shape, {"N": 0.09, "M": 0.11}.get(shape, 0.2) * tempo))
    keys.append(("M", 0.25))

    times, values = [], []
    now = 0.0
    for shape, duration in keys:
        opening, ratio, rounding = lip_reading._PROTOTYPES[SHAPES.index(shape)]
        target = (opening * open_scale, ratio, rounding * round_scale)
        times += [now, now + duration * 0.6]
        values += [target, target]
        now += duration
    values = np.array(values)

    mouths = []
    for t in np.arange(0, now, 1.0 / fps):
        opening, ratio, rounding = (
            np.interp(t, times, values[:, column]) for column in range(3)
        )
        mouths.append(
            MouthMeasure(
                max(0.0, float(opening + rng.normal(0, 0.012))),
                float(ratio + rng.normal(0, 0.03)) * DEFAULT_NEUTRAL_WIDTH,
                float(np.clip(rounding + rng.normal(0, 0.05), 0, 1)),
            )
        )
    return mouths


def test_simulated_speakers_get_the_right_word_or_can_pick_it():
    rng = np.random.default_rng(23)
    written_right = written_wrong = one_tap_away = total = 0

    for word in VOCABULARY:
        for _ in range(6):
            reading = read_word(_say(word, rng))
            total += 1
            written_right += reading.word == word
            written_wrong += reading.word not in (None, word)
            one_tap_away += reading.word == word or word in reading.alternatives

    # Lo importante es no escribir por la persona algo que no dijo y que la
    # palabra quede siempre a mano aunque el lector dude.
    assert written_right / total >= 0.5
    assert written_wrong / total <= 0.1
    assert one_tap_away / total >= 0.9


def test_a_word_said_slowly_is_written():
    rng = np.random.default_rng(1)

    reading = read_word(_say("ayuda", rng, speaker=False, tempo=2.0))

    assert reading.word == "ayuda"
    assert reading.reason is None
    assert "A" in reading.shapes and "U" in reading.shapes
    # Se ve la forma de la boca, sin el reposo de antes y después.
    assert reading.shapes[0] not in ("M", "N")
    assert reading.shapes[-1] not in ("M", "N")


def test_when_two_words_are_too_close_nothing_is_written(monkeypatch):
    monkeypatch.setattr(lip_reading, "_MIN_MARGIN", 100.0)
    rng = np.random.default_rng(1)

    reading = read_word(_say("ayuda", rng, speaker=False, tempo=2.0))

    assert reading.word is None
    # La mejor encabeza las candidatas, para que la persona elija.
    assert reading.alternatives[0] == "ayuda"
    assert 1 <= len(reading.alternatives) <= 3


def test_when_nothing_fits_well_nothing_is_written(monkeypatch):
    monkeypatch.setattr(lip_reading, "_MAX_COST", 0.0)
    rng = np.random.default_rng(1)

    assert read_word(_say("hola", rng, speaker=False)).word is None


def test_closing_the_mouth_from_an_a_does_not_invent_an_e():
    # Medido con MediaPipe sobre una foto real: abrir, quedarse en A y cerrar.
    # Al cerrar se pasa por algo con forma de E, y eso no es decir «clase».
    openings = [0.006, 0.006, 0.09, 0.19, 0.31, 0.347, 0.347]
    openings = openings + [0.347, 0.347, 0.31, 0.19, 0.09, 0.006, 0.006]
    widths = [0.597, 0.597, 0.62, 0.646, 0.645, 0.638, 0.638]
    widths = widths + [0.638, 0.638, 0.645, 0.646, 0.62, 0.597, 0.597]
    mouths = [MouthMeasure(o, w) for o, w in zip(openings, widths)]

    reading = read_word(mouths)

    assert reading.word != "clase"
    assert "mal" in (reading.word, *reading.alternatives)


def test_a_mouth_that_is_moving_is_less_still_than_one_that_stopped():
    mouths = [MouthMeasure(o, 0.6) for o in (0.0, 0.0, 0.3, 0.3, 0.3)]

    stillness = lip_reading._stillness(mouths, DEFAULT_NEUTRAL_WIDTH)

    assert stillness[4] == pytest.approx(1.0)
    assert stillness[1] < stillness[4]
    assert stillness.min() >= 0.0


def test_exaggerating_the_mouth_does_not_change_the_word():
    rng = np.random.default_rng(1)
    clear = _say("clase", rng, speaker=False)
    wide = [MouthMeasure(m.opening * 1.6, m.width, m.rounding) for m in clear]

    assert read_word(wide).word == read_word(clear).word


def test_a_word_too_long_for_the_clip_cannot_be_the_answer():
    # Tres fotos no alcanzan para las muchas formas de «imprimir».
    mouths = [_shape("M"), _shape("A"), _shape("M")]

    reading = read_word(mouths)

    assert "imprimir" not in (reading.word, *reading.alternatives)


def test_a_clip_too_short_is_not_read():
    assert read_word([_shape("A"), _shape("M")]).reason == REASON_TOO_SHORT


def test_a_clip_with_few_faces_is_not_read():
    mouths = [_shape("A"), _shape("M"), _shape("O")]

    reading = read_word(mouths, total_frames=10)

    assert reading.reason == REASON_NO_FACE
    assert reading.word is None
    assert reading.frames == 10
    assert reading.frames_with_face == 3


def test_a_still_mouth_says_nothing():
    reading = read_word([_shape("M")] * 12)

    assert reading.reason == REASON_NO_MOVEMENT
    assert reading.word is None
    assert reading.alternatives == ()


def test_shapes_seen_ignores_a_single_passing_frame_in_long_clips():
    seen = lip_reading._shapes_seen(list("MMAAAEOOOMMM"))

    assert seen == ("A", "O")


def test_shapes_seen_keeps_every_frame_in_short_clips():
    assert lip_reading._shapes_seen(list("MAEO")) == ("A", "E", "O")


def test_a_word_with_every_shape_hidden_only_answers_to_the_rest_of_the_clip():
    # «la» sería N-A; una palabra solo de consonantes ocultas no exige nada.
    scores = np.array([shape_scores(_shape("N"))] * 4)

    assert lip_reading._word_cost(scores, ["N"], np.ones(4)) < 1.0


def test_alignment_fails_when_there_are_fewer_frames_than_shapes():
    costs = np.zeros((1, 4))

    assert lip_reading._align(costs, [False, True, True, False]) is None


def test_alignment_keeps_the_order_of_the_shapes():
    # Estados: reposo, A, O, reposo. Fotos: O y luego A, al revés de la palabra.
    rows, required = lip_reading._states_of(["A", "O"])
    scores = np.array([shape_scores(_shape("O")), shape_scores(_shape("A"))])
    costs = 1.0 - scores @ rows.T

    path = lip_reading._align(costs, required)

    assert path == [1, 2]
    # Queda asignado, pero muy caro: nada lo deja pasar por una palabra buena.
    assert costs[0, 1] > 0.5 and costs[1, 2] > 0.5


def test_a_dead_end_in_the_alignment_is_not_an_answer():
    costs = np.full((2, 3), np.inf)

    assert lip_reading._align(costs, [False, True, False]) is None


def test_alignment_without_required_shapes_can_stay_at_rest():
    costs = np.zeros((2, 3))

    assert lip_reading._align(costs, [False, False, False]) is not None
