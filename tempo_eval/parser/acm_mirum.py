import csv
import sys
from os.path import join, basename, exists

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, create_tempo_annotation, get_bibtex_entry
from tempo_eval.evaluation import get_references_path

ACM_MIRUM = 'acm_mirum'


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_percival2014(input_audio_dir)
    parse_peeters2012(input_audio_dir)


def parse_percival2014(input_audio_dir):
    output_dir = get_references_path('acm_mirum', 'jams')
    smkdirs(output_dir)
    input_annotation_file = get_references_path('acm_mirum', 'percival2014', 'acm_mirum_tempos.mf')
    with open(input_annotation_file, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            filename = basename(row[0])
            bpm = float(row[1])
            jams_file = join(output_dir, filename.replace('.wav', '.jams'))
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, input_audio_dir)
            jam.annotations.append(_create_percical2014_tempo_annotation(bpm))
            jam.save(jams_file)


def parse_peeters2012(input_audio_dir):
    output_dir = get_references_path('acm_mirum', 'jams')
    input_annotation_file = get_references_path('acm_mirum', 'peeters2012', 'acm-mirum2012-testset.txt')
    with open(input_annotation_file, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if row[0].startswith('REF'):
                continue
            full_name = row[0]
            splits = full_name.split('-')
            name = splits[len(splits)-1] + '.clip'
            filename = name + '.wav'
            jams_file = join(output_dir, name + '.jams')
            bpm = float(row[1])

            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, input_audio_dir)

            jam.file_metadata.identifiers.update(item_7digital_id=full_name)
            jam.annotations.append(_create_peeters2012_tempo_annotation(bpm))
            jam.save(jams_file)


def _create_percical2014_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=ACM_MIRUM,
        version='2.0',
        curator=jams.Curator(name='Graham Percival', email='graham@percival-music.ca'),
        data_source='',
        annotator={'bibtex': get_bibtex_entry('Percival2014'),
                   'ref_url': 'http://www.marsyas.info/tempo/'})
    return tempo


def _create_peeters2012_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=ACM_MIRUM,
        version='1.0',
        curator=jams.Curator(name='Geoffroy Peeters', email='geoffroy.peeters@telecom-paristech.fr'),
        data_source='',
        annotator={'bibtex': get_bibtex_entry('Peeters2012'),
                   'ref_url': 'http://recherche.ircam.fr/anasyn/peeters/pub/2012_ACMMIRUM/'})
    return tempo


if __name__ == '__main__':
    parse(*sys.argv[1:])