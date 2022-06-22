import logging
import sys
from os.path import join, exists
from statistics import median

import jams
import numpy as np
import pandas as pd
from jams.util import smkdirs
from scipy.io import loadmat

from tempo_eval.parser.util import create_jam, timestamps_to_bpm, create_tempo_annotation, \
    get_bibtex_entry, create_tag_open_annotation, create_beat_annotation, inter_timestamp_intervals
from tempo_eval.evaluation import get_references_path

KLAPURI = 'klapuri'


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_klapuri2006(input_audio_dir)


def parse_klapuri2006(input_audio_dir):

    output_dir = get_references_path('klapuri', 'jams')
    smkdirs(output_dir)

    # periodRefs.mat, provided by Anssi Klapuri
    annotation_file = get_references_path('klapuri', 'klapuri2006', 'periodRefs.mat')
    # database.html, scraped from
    # https://web.archive.org/web/20081117040557/http://www.cs.tut.fi/~klap/iiro/meter/database.html
    metadata_file = get_references_path('klapuri', 'klapuri2006', 'database.html')
    # title_filename_map.tsv, to connect the two files above
    mapping_file = get_references_path('klapuri', 'klapuri2006', 'title_filename_map.tsv')
    # EULA from TUT
    eula_file = get_references_path('klapuri', 'klapuri2006', 'EULA.txt')

    with open(eula_file, mode='r') as f:
        eula = f.read()

    mapping = pd.read_csv(mapping_file, sep='\t', names=['title', 'file'])
    annotation_file = loadmat(annotation_file, uint16_codec='utf-16')
    period_refs = annotation_file['periodRefs']

    # TODO: Is row[3] really meter?
    def fix_meter(m):
        return m[0] if m else 0

    annotations = pd.DataFrame(
        {
            'tactus': [np.concatenate(row[0]).ravel().tolist() for row in period_refs],
            'measure': [np.concatenate(row[1]).ravel().tolist() for row in period_refs],
            'tatum': [np.concatenate(row[2]).ravel().tolist() for row in period_refs],
            'tatum_per_tactus': [fix_meter(row[3][0]) for row in period_refs],
            'file': [row[4][0].replace('.ann', '') for row in period_refs],
            # stability: informal annotation of the "stability" of the tempo/beat:
            #  0: stable tempo,
            # -1: break (discontinuity/change in tempo),
            # -2: expression (tempo variation),
            # -3: both.
            'stability': [np.concatenate(row[5]).ravel().tolist() for row in period_refs],
        }
    )

    tables = pd.read_html(metadata_file)
    # the third table is the one we want
    df = tables[2]
    # drop first two and last rows (bad data)
    df.drop([0, 1, 476], inplace=True)
    # fix column names
    df.columns = ['artist', 'title', 'genre', 'tatum', 'tactus', 'measure']
    # fix nan values in artist column
    df.artist = df.artist.transform(lambda x: x if isinstance(x, str) else '')

    # not needed, as this data is already in the annotations
    del df['tatum']
    del df['tactus']
    del df['measure']

    # normalize genre names
    df.genre = df.genre.transform(lambda x: x.lower())
    # the HTML contains duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    # join annotations with mapping
    df = df.merge(mapping, on='title')
    df = df.merge(annotations, on='file')

    # TODO: Manually check El Choclo

    for _, row in df.iterrows():
        jams_file = join(output_dir, row.file + '.jams')
        audio_file = row.file + '.wav'
        if exists(jams_file):
            logging.warning('Adding to an existing jams file: {}, {}'.format(jams_file, row))
            jam = jams.load(jams_file)
        else:
            jam = create_jam(audio_file, input_audio_dir, artist=row.artist, title=row.title)

        beats_per_measure = estimate_beats_per_measure(row.tactus, row.measure)
        if row.tactus:
            median_bpm, _, _, _ = timestamps_to_bpm(row.tactus, beats_per_measure)
            jam.annotations.append(_create_klapuri2006_tempo_annotation(median_bpm, eula))
            derived_beats = derive_beats(row.tactus, row.measure, beats_per_measure)
            jam.annotations.append(_create_klapuri2006_beat_annotation(derived_beats, eula))

        # add genre and tempo variation tags
        tags = derive_tags(beats_per_measure, row.genre, row.stability)
        jam.annotations.append(_create_klapuri2006_tag_open_annotation(tags, eula))

        # bogus duration value to please jams
        if jam.file_metadata.duration is None:
            jam.file_metadata.duration = 99999
        # add file identifier, because we did so for others
        if not jam.file_metadata.identifiers:
            jam.file_metadata.identifiers = {'file': audio_file}
        jam.save(jams_file)


def derive_tags(beats_per_measure, genre, stability):
    tags = [genre]
    if beats_per_measure > 1:
        tags.append('{} beats/measure'.format(beats_per_measure))
    else:
        tags.append('unkown meter')
    for s in stability:
        if s == -1 or s == -3:
            tags.append('tempo discontinuity')
        elif s == -2 or s == -3:
            tags.append('tempo variation')
    return tags


def derive_beats(beats, measures, beats_per_measure):
    beat_dict = []
    beat_position = -1
    measure_number = 0
    tolerance = median(inter_timestamp_intervals(beats)) / 5.
    for tactus in beats:
        if len(measures) > measure_number and (measures[measure_number] - tolerance) < tactus:
            beat_position = 1
            measure_number += 1
        if beat_position > beats_per_measure > 1:
            confidence = 0.5
        else:
            confidence = 1.
        beat_dict.append({'timestamp': tactus, 'position': beat_position, 'confidence': confidence})
        if beat_position > 0:
            beat_position += 1
    return beat_dict


def estimate_beats_per_measure(beats, measures):
    if beats and measures:
        median_tactus_diff = median(inter_timestamp_intervals(beats))
        median_measure_diff = median(inter_timestamp_intervals(measures))
        beats_per_measure = round(median_measure_diff / median_tactus_diff)
    else:
        beats_per_measure = 1
    return beats_per_measure


def _create_klapuri2006_tempo_annotation(bpm, license):
    tempo = create_tempo_annotation(bpm, license=license)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=KLAPURI,
        version='1.0',
        curator=jams.Curator(name='Anssi Klapuri', email='anssi@yousician.com'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Klapuri2006')})
    return tempo


def _create_klapuri2006_tag_open_annotation(tags, license):
    tag = create_tag_open_annotation(*tags, license=license)
    tag.annotation_metadata = jams.AnnotationMetadata(
        corpus=KLAPURI,
        version='1.0',
        curator=jams.Curator(name='Anssi Klapuri', email='anssi@yousician.com'),
        data_source='manual annotation',
        annotator={'bibtex': get_bibtex_entry('Klapuri2006')})
    return tag


def _create_klapuri2006_beat_annotation(beats, license):
    beat = create_beat_annotation(beats, license=license)
    beat.annotation_metadata = jams.AnnotationMetadata(
        corpus=KLAPURI,
        version='1.0',
        curator=jams.Curator(name='Anssi Klapuri', email='anssi@yousician.com'),
        data_source='manual annotation',
        annotation_rules='Beat positions automatically derived from measure timestamps. '
                         'Confidence refers to beat value, not timestamp.',
        annotator={'bibtex': get_bibtex_entry('Klapuri2006')})
    return beat


if __name__ == '__main__':
    parse(*sys.argv[1:])
