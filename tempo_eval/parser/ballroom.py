import csv
import sys
from os import walk
from os.path import join, exists, basename, dirname

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, timestamps_to_bpm, create_tempo_annotation, \
    create_beat_annotation, create_tag_open_annotation, get_bibtex_entry
from tempo_eval.evaluation import get_references_path

BALLROOM = 'ballroom'


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_ismir2004(input_audio_dir)
    parse_krebs2013(input_audio_dir)
    parse_percival2014(input_audio_dir)


def parse_ismir2004(input_audio_dir):
    output_dir = get_references_path('ballroom', 'jams')
    smkdirs(output_dir)
    input_annotation_dir = get_references_path('ballroom', 'ismir2004')

    for (dirpath, _, filenames) in walk(input_annotation_dir):
        for filename in [f for f in filenames if f.endswith('.bpm')]:
            jams_file = join(output_dir, filename.replace('.bpm', '.jams'))
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, input_audio_dir)
            with open(join(dirpath, filename), mode='r') as file:
                bpm = float(file.read())
                jam.annotations.append(_create_ismir2004_tempo_annotation(bpm))
                jam.save(jams_file)


def parse_krebs2013(input_audio_dir):
    output_dir = get_references_path('ballroom', 'jams')
    input_annotation_dir = get_references_path('ballroom', 'krebs2013')

    for (dirpath, _, filenames) in walk(input_annotation_dir):
        for filename in [f for f in filenames if f.endswith('.beats')]:
            jams_file = join(output_dir, filename.replace('.beats', '.jams'))
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, input_audio_dir)
            with open(join(dirpath, filename), mode='r') as file:
                lines = file.readlines()
                beats = []
                for line in lines:
                    splits = line.split()
                    beats.append({'timestamp': float(splits[0]), 'position': int(splits[1]), 'confidence': 1.0})
                meter = max([b['position'] for b in beats])
                timestamps = [b['timestamp'] for b in beats]
                median_bpm, _, _, cv = timestamps_to_bpm(timestamps, meter=meter)
                jam.annotations.append(_create_krebs2013_tempo_annotation(median_bpm))
                jam.annotations.append(_create_krebs2013_beat_annotation(beats))
                jam.save(jams_file)


def parse_percival2014(input_audio_dir):
    output_dir = get_references_path('ballroom', 'jams')
    input_annotation_file = get_references_path('ballroom', 'percival2014', 'ballroom_tempos.mf')
    with open(input_annotation_file, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            genre = basename(dirname(row[0]))
            filename = basename(row[0])
            bpm = float(row[1])
            jams_file = join(output_dir, filename.replace('.wav', '.jams'))
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, input_audio_dir)
            jam.annotations.append(_create_percival2014_annotation(bpm))
            jam.annotations.append(_create_ismir2004_genre_annotation(genre))
            jam.save(jams_file)


def _create_ismir2004_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=BALLROOM,
        version='1.0',
        curator=jams.Curator(name='Simon Dixon', email='s.e.dixon@qmul.ac.uk'),
        data_source='BallroomDancers.com, checked by human',
        annotator={'bibtex': get_bibtex_entry('Gouyon2006'),
                   'ref_url': 'http://mtg.upf.edu//ismir2004_auco/contest/tempoContest/node5.html'})
    return tempo


def _create_percival2014_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=BALLROOM,
        version='3.0',
        curator=jams.Curator(name='Graham Percival', email='graham@percival-music.ca'),
        data_source='BallroomDancers.com, checked by human',
        annotator={'bibtex': get_bibtex_entry('Percival2014'),
                   'ref_url': 'http://www.marsyas.info/tempo/'})
    return tempo


def _create_krebs2013_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=BALLROOM,
        version='2.0',
        curator=jams.Curator(name='Florian Krebs', email='florian.krebs@jku.at'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of corresponding inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Krebs2013'),
                   'ref_url': 'https://github.com/CPJKU/BallroomAnnotations'})
    return tempo


def _create_krebs2013_beat_annotation(beats):
    beat = create_beat_annotation(beats)
    beat.annotation_metadata = jams.AnnotationMetadata(
        corpus=BALLROOM,
        version='1.0',
        curator=jams.Curator(name='Florian Krebs', email='florian.krebs@jku.at'),
        data_source='manual annotation',
        annotator={'bibtex': get_bibtex_entry('Krebs2013'),
                   'ref_url': 'https://github.com/CPJKU/BallroomAnnotations'})
    return beat


def _create_ismir2004_genre_annotation(genre):
    tag = create_tag_open_annotation(genre)
    tag.annotation_metadata = jams.AnnotationMetadata(
        corpus=BALLROOM,
        version='1.0',
        curator=jams.Curator(name='Simon Dixon', email='s.e.dixon@qmul.ac.uk'),
        data_source='BallroomDancers.com',
        annotator={'bibtex': get_bibtex_entry('Gouyon2006'),
                   'ref_url': 'http://mtg.upf.edu//ismir2004_auco/contest/tempoContest/node5.html'})
    return tag


if __name__ == '__main__':
    parse(*sys.argv[1:])