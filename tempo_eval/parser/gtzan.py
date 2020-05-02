import csv
import sys
from os import walk
from os.path import join, basename, dirname, exists

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, timestamps_to_bpm, create_tempo_annotation, \
    get_bibtex_entry
from tempo_eval.evaluation import get_references_path

GTZAN = 'GTZAN'


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_marchand2015()
    parse_percival2014(input_audio_dir)
    parse_tzanetakis2013(input_audio_dir)


def parse_marchand2015():
    output_dir = get_references_path('gtzan', 'jams')
    smkdirs(output_dir)
    input_annotation_dir = get_references_path('gtzan', 'marchand2015')

    for (dirpath, _, filenames) in walk(input_annotation_dir):
        for filename in [f for f in filenames if f.endswith('.jams')]:
            jams_file = join(output_dir, filename.replace('.wav', ''))
            jam = jams.load(join(dirpath, filename))

            sandbox1 = jam.annotations['tag_open'][1]['sandbox']
            tags = []

            # identify meter, so find corresponding beats
            meter = 1
            if 'meter' in sandbox1 and '/' in sandbox1['meter']:
                meter = int(sandbox1['meter'].split('/')[0])
                tags.append(sandbox1['meter'])

            # iterate over beats
            beats = jam.annotations['beat'][0]['data']
            timestamps = [b.time for b in beats]

            # ICBI-based
            median_bpm, _, std, cv = timestamps_to_bpm(timestamps, meter)
            jam.annotations.append(_create_marchand2015_median_cor_beat_tempo_annotation(median_bpm))

            # IBI-based
            median_bpm, _, std, cv = timestamps_to_bpm(timestamps)
            jam.annotations.append(_create_marchand2015_median_beat_tempo_annotation(median_bpm))

            # derive some additional tags from first sandbox
            sandbox0 = jam.annotations['tag_open'][0]['sandbox']
            for k in sandbox0.keys():
                if 'yes' == sandbox0[k]:
                    tags.append(k)
                else:
                    tags.append('no_' + k)

            # simply append tags to first 'tag_open' annotation, so that we can later
            # process them easily
            for tag in set(tags):
                if len(tag.strip()) > 0:
                    jam.annotations['tag_open'][0].append(time=0.0, value=tag, duration='nan', confidence=1.0)

            jam.save(jams_file)


def parse_percival2014(input_audio_dir):
    output_dir = get_references_path('gtzan', 'jams')
    input_annotation_file = get_references_path('gtzan', 'percival2014', 'genres_tempos.mf')

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


def parse_tzanetakis2013(input_audio_dir):
    output_dir = get_references_path('gtzan', 'jams')
    input_annotation_file = get_references_path('gtzan', 'tzanetakis2013', 'genres_tempos.mf')

    with open(input_annotation_file, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            genre = basename(dirname(row[0])).replace('hiphop', 'hip-hop')  # to please tag_gtzan namespace
            filename = basename(row[0])
            bpm = float(row[1])
            jams_file = join(output_dir, filename.replace('.wav', '.jams'))
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, input_audio_dir)
            jam.annotations.append(_create_tzanetakis2013_tempo_annotation(bpm))
            jam.annotations.append(_create_tzanetakis2013_genre_annotation(genre))
            jam.save(jams_file)


def _create_percical2014_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=GTZAN,
        version='2.0',
        curator=jams.Curator(name='Graham Percival', email='graham@percival-music.ca'),
        data_source='',
        annotator={'bibtex': get_bibtex_entry('Percival2014'),
                   'ref_url': 'http://www.marsyas.info/tempo/'})
    return tempo


def _create_tzanetakis2013_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=GTZAN,
        version='1.0',
        curator=jams.Curator(name='George Tzanetakis', email='gtzan@cs.uvic.ca'),
        data_source='',
        annotator={'bibtex': get_bibtex_entry('Tzanetakis2013'),
                   'ref_url': 'http://www.marsyas.info/tempo/'})
    return tempo


def _create_tzanetakis2013_genre_annotation(genre):
    tag = jams.Annotation(namespace='tag_gtzan')
    tag.annotation_metadata = jams.AnnotationMetadata(
        corpus=GTZAN,
        version='1.0',
        curator=jams.Curator(name='George Tzanetakis', email='gtzan@cs.uvic.ca'),
        data_source='',
        annotator={'bibtex': get_bibtex_entry('Tzanetakis2002'),
                   'ref_url': 'http://www.marsyas.info/tempo/'})
    tag.append(time=0.0, value=genre, duration='nan', confidence=1.0)
    return tag


def _create_marchand2015_median_beat_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=GTZAN,
        version='GTZAN-Rhythm_v2_ismir2015_lbd_2015-10-28_IBI',
        curator=jams.Curator(name='Ugo Marchand & Quentin Fresnel', email='ugo.marchand@ircam.fr'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of inter beat intervals (IBI)',
        annotator={'bibtex': get_bibtex_entry('Marchand2015'),
                   'ref_url': 'https://hal.archives-ouvertes.fr/hal-01252603/document'})
    return tempo


def _create_marchand2015_median_cor_beat_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=GTZAN,
        version='GTZAN-Rhythm_v2_ismir2015_lbd_2015-10-28_ICBI',
        curator=jams.Curator(name='Ugo Marchand & Quentin Fresnel', email='ugo.marchand@ircam.fr'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of inter corresponding beat intervals (ICBI)',
        annotator={'bibtex': get_bibtex_entry('Marchand2015'),
                   'ref_url': 'https://hal.archives-ouvertes.fr/hal-01252603/document'})
    return tempo


if __name__ == '__main__':
    parse(*sys.argv[1:])