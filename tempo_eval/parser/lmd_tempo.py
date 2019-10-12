import csv
import os
import sys
from os.path import join, exists

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, create_tempo_annotation, \
    create_tag_open_annotation, get_bibtex_entry
from tempo_eval.evaluation import get_references_path

LMD_TEMPO = 'lmd_tempo'


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_schreiber2018(input_audio_dir)


def parse_schreiber2018(input_audio_dir):
    output_dir = get_references_path('lmd_tempo', 'jams')
    smkdirs(output_dir)
    input_annotation_file = get_references_path('lmd_tempo', 'schreiber2018', 'lmd_tempo.tsv')

    with open(input_annotation_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            msd_id = row[0]
            bpm = float(row[1])
            genre = row[3]
            jams_file = join(output_dir, msd_id + '.jams')
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                # create full msd path
                preview_file_name = msd_id + '.mp3'
                jam = create_jam(preview_file_name, join(input_audio_dir, msd_id[2] + os.sep + msd_id[3] + os.sep + msd_id[4]))
            jam.annotations.append(_create_schreiber2018_tempo_annotation(bpm))
            if genre != 'unknown':
                jam.annotations.append(_create_schreiber2018_genre_annotation(genre))
            # inject MSD id
            jam.file_metadata.identifiers.update(msd_id=msd_id)
            jam.save(jams_file)


def _create_schreiber2018_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=LMD_TEMPO,
        version='1.0',
        curator=jams.Curator(name='Hendrik Schreiber', email='hs@tagtraum.com'),
        data_source='LMD 0.1, https://colinraffel.com/projects/lmd/',
        annotation_rules='Consensus between Schreiber2017 and MIDI messages (2% tolerance).',
        annotation_tools='Schreiber2017, MIDI parser',
        annotator={'bibtex': get_bibtex_entry('Schreiber2018a'),
                   'ref_url': 'http://www.tagtraum.com/tempo_estimation.html'})
    return tempo


def _create_schreiber2018_genre_annotation(genre):
    tag = create_tag_open_annotation(genre)
    tag.annotation_metadata = jams.AnnotationMetadata(
        corpus=LMD_TEMPO,
        version='1.0',
        curator=jams.Curator(name='Hendrik Schreiber', email='hs@tagtraum.com'),
        data_source='LMD 0.1, https://colinraffel.com/projects/lmd/',
        annotation_rules='Intersection with Schreiber2015 MSD genre annotations.',
        annotator={'bibtex': get_bibtex_entry('Schreiber2018a'),
                   'ref_url': 'http://www.tagtraum.com/tempo_estimation.html'})
    return tag


if __name__ == '__main__':
    parse(*sys.argv[1:])