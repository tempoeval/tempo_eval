import sys
from os.path import join, exists
from statistics import median

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, timestamps_to_bpm, create_tempo_annotation, \
    create_beat_annotation, create_tag_open_annotation, get_bibtex_entry
from tempo_eval.evaluation import get_references_path

CORPUS = 'hainsworth'
CURATOR = jams.Curator(name='Stephen Webley Hainsworth', email='swh21@eng.cam.ac.uk')


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_hainsworth2004(input_audio_dir)


def parse_hainsworth2004(input_audio_dir):
    output_dir = get_references_path('hainsworth', 'jams')
    smkdirs(output_dir)
    input_annotation_file = get_references_path('hainsworth', 'hainsworth2004', 'hainsworth.txt')

    with open(input_annotation_file, mode='r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            columns = line.split('<sep>')
            filename = columns[0].strip()
            artist = columns[1].strip()
            title = columns[2].strip()
            style = columns[4].strip()
            big_style = columns[5].strip()
            tempo_cond = columns[6].strip()
            sub_div = columns[8].strip()
            original_bpm = float(columns[9].strip())
            # for some reason Hainsworth stored beat timestamps
            # not in seconds, but as sample number
            timestamps = [float(b)/44100.0 for b in columns[10].strip().split(',')]
            downbeat_indices = [int(b) for b in columns[11].strip().split(',')]
            median_bpm, _, _, _ = timestamps_to_bpm(timestamps, meter=1)
            meter = _derive_meter(downbeat_indices)
            beats = _derive_beats(timestamps, meter, downbeat_indices)
            corresponding_median_bpm, _, _, cv = timestamps_to_bpm(timestamps, meter=meter)

            jams_file = join(output_dir, filename.replace('.wav', '.jams'))
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, input_audio_dir, artist=artist, title=title)
            jam.annotations.append(_create_hainsworth2004_original_tempo_annotation(original_bpm))
            jam.annotations.append(_create_hainsworth2004_median_cor_beat_tempo_annotation(corresponding_median_bpm))
            jam.annotations.append(_create_hainsworth2004_median_beat_tempo_annotation(median_bpm))
            jam.annotations.append(_create_hainsworth2004_beat_annotation(beats))
            jam.annotations.append(_create_hainsworth2004_style_annotation(style, big_style, tempo_cond, sub_div))
            jam.save(jams_file)


def _derive_beats(timestamps, meter, downbeat_indices):
    beat_positions = []
    position = int((meter - downbeat_indices[0] + 1) % meter)
    for i in range(len(timestamps)):
        beat_positions.append(position + 1)
        position = ((position + 1) % meter)
    return [{'timestamp': ts, 'position': pos, 'confidence': 1.0} for ts, pos in zip(timestamps, beat_positions)]


def _derive_meter(downbeat_indices):
    index_diffs = []
    for i in range(len(downbeat_indices) - 1):
        index_diffs.append(downbeat_indices[i + 1] - downbeat_indices[i])
    return int(median(index_diffs))


def _create_hainsworth2004_original_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=CORPUS,
        version='1.0',
        curator=CURATOR,
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='mean of inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Hainsworth2004')})
    return tempo


def _create_hainsworth2004_median_cor_beat_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=CORPUS,
        version='2.0',
        curator=CURATOR,
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of corresponding inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Hainsworth2004')})
    return tempo


def _create_hainsworth2004_median_beat_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=CORPUS,
        version='3.0',
        curator=CURATOR,
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Hainsworth2004')})
    return tempo


def _create_hainsworth2004_beat_annotation(beats):
    beat = create_beat_annotation(beats)
    beat.annotation_metadata = jams.AnnotationMetadata(
        corpus=CORPUS,
        version='1.0',
        curator=CURATOR,
        data_source='manual annotation',
        annotator={'bibtex': get_bibtex_entry('Hainsworth2004')})
    return beat


def _create_hainsworth2004_style_annotation(style, big_style, tempo_cond, sub_div):
    tag = create_tag_open_annotation(style,
                                     big_style,
                                     # remove redundant info for simplicity
                                     tempo_cond.replace('none', '').replace('(Rub)', '').replace('(Rall)', '').replace('(SD)', '').strip(),
                                     sub_div)
    tag.annotation_metadata = jams.AnnotationMetadata(
        corpus=CORPUS,
        version='1.0',
        curator=CURATOR,
        data_source='manual annotation',
        annotator={'bibtex': get_bibtex_entry('Hainsworth2004')})
    return tag


if __name__ == '__main__':
    parse(*sys.argv[1:])
