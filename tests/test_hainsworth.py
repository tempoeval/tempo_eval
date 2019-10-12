from tempo_eval.parser.hainsworth import _derive_beats, _derive_meter


def test_derive_beats():
    timestamps = [20064,46050,71272,97640,124678,152018,178895,204642,231774,258143,284129]

    downbeat_indices = [4,8,12]
    beats = _derive_beats(timestamps, 4, downbeat_indices)
    beat_positions = [b['position'] for b in beats]
    assert beat_positions == [2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

    downbeat_indices = [1,5,9]
    beats = _derive_beats(timestamps, 4, downbeat_indices)
    beat_positions = [b['position'] for b in beats]
    assert beat_positions == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]

    downbeat_indices = [3,6,9]
    beats = _derive_beats(timestamps, 3, downbeat_indices)
    beat_positions = [b['position'] for b in beats]
    assert beat_positions == [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_derive_meter():
    assert _derive_meter([1, 4, 7]) == 3
    assert _derive_meter([4, 8, 12]) == 4
