import csv
import sys
from os.path import join, exists

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, create_tempo_annotation, \
    create_tag_open_annotation, get_bibtex_entry
from tempo_eval.evaluation import get_references_path

GIANTSTEPS_MTG_KEY = 'GiantSteps MTG Key'


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_schreiber2018(input_audio_dir)


def parse_schreiber2018(input_audio_dir):
    output_dir = get_references_path('giantsteps_mtg_key', 'jams')
    smkdirs(output_dir)
    input_annotation_file = get_references_path('giantsteps_mtg_key', 'schreiber2018', 'giantsteps_mtg_tempo.tsv')
    meta_data_file = get_references_path('giantsteps_mtg_key', 'schreiber2018', 'beatport_metadata.txt')
    meta_data = {}
    with open(meta_data_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            meta_data[row['ID']] = row

    with open(input_annotation_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            beatport_id = row[0]
            bpm = float(row[1])
            key = row[2].replace('m', ':minor')
            if ':' not in key:
                key += ':major'
            genre = row[3]

            if bpm == 0.0:
                continue

            jams_file = join(output_dir, beatport_id + '.LOFI.jams')
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                # create full msd path
                preview_file_name = beatport_id + '.LOFI.mp3'
                artist = meta_data[beatport_id]['ARTIST']
                title = meta_data[beatport_id]['SONG TITLE']
                release = meta_data[beatport_id]['MIX'] + ', ' + meta_data[beatport_id]['LABEL']
                jam = create_jam(preview_file_name, input_audio_dir, artist=artist, title=title, release=release)
            jam.annotations.append(_create_schreiber2018_tempo_annotation(bpm))
            jam.annotations.append(_create_faraldo2017_key_annotation(key))
            jam.annotations.append(_create_faraldo2017_genre_annotation(genre))
            # inject beatport id
            jam.file_metadata.identifiers.update(beatport_id=beatport_id)
            jam.save(jams_file)


def _create_schreiber2018_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=GIANTSTEPS_MTG_KEY,
        version='1.0',
        curator=jams.Curator(name='Hendrik Schreiber', email='hs@tagtraum.com'),
        annotation_tools='manual annotation',
        annotator={'bibtex': get_bibtex_entry('Schreiber2018a'),
                   'ref_url': 'http://www.tagtraum.com/tempo_estimation.html'})
    return tempo


def _create_faraldo2017_genre_annotation(genre):
    tag = create_tag_open_annotation(genre)
    tag.annotation_metadata = jams.AnnotationMetadata(
        corpus=GIANTSTEPS_MTG_KEY,
        version='1.0',
        curator=jams.Curator(name='Ángel Faraldo', email='angelfaraldo@gmail.com'),
        annotation_tools="BeatPort import",
        annotator={'bibtex': get_bibtex_entry('Faraldo2017'),
                   'ref_url': 'https://github.com/GiantSteps/giantsteps-mtg-key-dataset'})
    return tag


def _create_faraldo2017_key_annotation(key):
    key_mode = jams.Annotation(namespace='key_mode')
    key_mode.annotation_metadata = jams.AnnotationMetadata(
        corpus=GIANTSTEPS_MTG_KEY,
        version='1.0',
        curator=jams.Curator(name='Ángel Faraldo', email='angelfaraldo@gmail.com'),
        annotator={'bibtex': get_bibtex_entry('Faraldo2017'),
                   'ref_url': 'https://github.com/GiantSteps/giantsteps-mtg-key-dataset'})
    key_mode.append(time=0.0,
                    duration='nan',
                    value=key,
                    confidence=1.0)
    return key_mode


if __name__ == '__main__':
    parse(*sys.argv[1:])
