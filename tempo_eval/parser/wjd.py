import csv
import sys
from os import walk
from os.path import join, exists

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, timestamps_to_bpm, create_tempo_annotation, \
    create_beat_annotation, get_bibtex_entry
from tempo_eval.evaluation import get_references_path

WJD = 'wjd'


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_pfleiderer2017(input_audio_dir)


def parse_pfleiderer2017(input_audio_dir):
    output_dir = get_references_path('wjd', 'jams')
    smkdirs(output_dir)
    input_annotation_dir = get_references_path('wjd', 'pfleiderer2017')
    for (dirpath, _, filenames) in walk(input_annotation_dir):
        for filename in [f for f in filenames if f.endswith('.csv')]:
            artist, title = _artist_title(filename)
            jams_file = join(output_dir, filename.replace('.csv', '.jams'))
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, input_audio_dir, artist=artist, title=title)
            with open(join(dirpath, filename), mode='r') as file:
                reader = csv.reader(file)
                beats = []
                meter = 1
                for row in reader:
                    # bar,bass_pitch,beat,chord,chorus_id,form,onset,signature
                    if row[0].startswith('bar'):
                        continue
                    position = int(row[2])
                    timestamp = float(row[6])
                    if row[7].find('/') > 0:
                        meter = int(row[7].split('/')[0])
                    beats.append({'timestamp': timestamp, 'position': position, 'confidence': 1.0})
                timestamps = [b['timestamp'] for b in beats]
                # dumb mean, not taking meter into account
                _, mean_bpm, _, _ = timestamps_to_bpm(timestamps, meter=1)
                # corresponding beats, median
                median_bpm, _, _, _ = timestamps_to_bpm(timestamps, meter=meter)

                jam.annotations.append(_create_pfleiderer2017_median_tempo_annotation(median_bpm))
                jam.annotations.append(_create_pfleiderer2017_mean_tempo_annotation(mean_bpm))
                jam.annotations.append(_create_pfleiderer2017_beat_annotation(beats))
                jam.save(jams_file)


def _artist_title(file_name):
    file_name = file_name.replace('_SOLO.csv', '')
    splits = file_name.split('_')
    artist = splits[0]
    title = splits[1]

    def space_before_upper(s):
        r = s[0]
        for c in s[1:]:
            if c.isupper():
                r += ' '
            r += c
        return r

    return space_before_upper(artist), space_before_upper(title)


def _create_pfleiderer2017_median_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=WJD,
        version='2.0',
        curator=jams.Curator(name='Hendrik Schreiber', email='hs@tagtraum.com'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of corresponding inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Pfleiderer2017'),
                   'ref_url': 'https://jazzomat.hfm-weimar.de'})
    return tempo


def _create_pfleiderer2017_mean_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=WJD,
        version='1.0',
        curator=jams.Curator(name='Martin Pfleiderer'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='mean of inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Pfleiderer2017'),
                   'ref_url': 'https://jazzomat.hfm-weimar.de'})
    return tempo


def _create_pfleiderer2017_beat_annotation(beats):
    beat = create_beat_annotation(beats)
    beat.annotation_metadata = jams.AnnotationMetadata(
        corpus=WJD,
        version='1.0',
        curator=jams.Curator(name='Martin Pfleiderer'),
        data_source='manual annotation',
        annotator={'bibtex': get_bibtex_entry('Pfleiderer2017'),
                   'ref_url': 'https://jazzomat.hfm-weimar.de'})
    return beat


if __name__ == '__main__':
    parse(*sys.argv[1:])