import sys
from os import walk
from os.path import join, exists, dirname, basename

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, timestamps_to_bpm, create_tempo_annotation, \
    create_beat_annotation, get_bibtex_entry
from tempo_eval.evaluation import get_references_path

BEATLES = 'beatles'


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_harte2010(input_audio_dir)


def parse_harte2010(input_audio_dir):
    output_dir = get_references_path('beatles', 'jams')
    input_annotation_dir = get_references_path('beatles', 'harte2010')

    for (dirpath, _, filenames) in walk(input_annotation_dir):
        for filename in [f for f in filenames if f.endswith('.txt')]:
            duration = _read_duration(dirpath, filename)
            artist, release, title = _artist_release_title(join(dirpath, filename))
            jams_file = join(output_dir, basename(dirpath), filename.replace('.txt', '.jams'))
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, input_audio_dir, artist=artist, title=title,
                                 release=release, duration=duration)

            with open(join(dirpath, filename), mode='r') as file:
                lines = file.readlines()
                beats = []
                for line in lines:
                    splits = line.split()
                    timestamp = float(splits[0])
                    position = 0
                    if splits[1].isdigit():
                        position = int(splits[1])
                    beats.append({'timestamp': timestamp, 'position': position, 'confidence': 0.8})
                # only look at the first couple of beats, as sometimes the last
                # beat in the track is a 5, even though we are in 4/4
                meter = max([b['position'] for b in beats[:20]])
                timestamps = [b['timestamp'] for b in beats]
                median_bpm, mean, std, cv = timestamps_to_bpm(timestamps, meter=meter)
                jam.annotations.append(_create_harte2010_tempo_annotation(median_bpm))
                jam.annotations.append(_create_harte2010_beat_annotation(beats))
                smkdirs(dirname(jams_file))
                jam.save(jams_file)


def _artist_release_title(file_name):
    file_name = file_name.replace('.txt', '').replace('_', ' ')
    splits = file_name.split('/')
    title = splits[-1].replace('CD1 - ', '').replace('CD2 - ', '')[5:]
    release = splits[-2].replace('CD1', '').replace('CD2', '')[5:]
    artist = 'The Beatles'
    return artist, release, title


def _read_duration(dirpath, filename):
    """
    Read duration from lab file.

    :param dirpath: path
    :param filename: .txt file name (beat annotations)
    :return: duration or ``None``, if not found
    """
    structure = join(dirpath, filename.replace('.txt', '.lab'))
    duration = None
    if exists(structure):
        with open(structure, mode='r') as file:
            lines = file.readlines()
            last_line = lines[-1]
            duration = float(last_line.split()[1])
    return duration


def _create_harte2010_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=BEATLES,
        version='1.2',
        curator=jams.Curator(name='Christopher Harte', email='chris@melodient.com'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of corresponding inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Harte2010'),
                   'ref_url': 'http://isophonics.net/content/reference-annotations-beatles'})
    return tempo


def _create_harte2010_beat_annotation(beats):
    beat = create_beat_annotation(beats)
    beat.annotation_metadata = jams.AnnotationMetadata(
        corpus=BEATLES,
        version='1.2',
        curator=jams.Curator(name='Christopher Harte', email='chris@melodient.com'),
        data_source='manual annotation',
        validation='Checked by Matthew Davies. Use with moderate confidence.',
        annotator={'bibtex': get_bibtex_entry('Harte2010'),
                   'ref_url': 'http://isophonics.net/content/reference-annotations-beatles'})
    return beat


if __name__ == '__main__':
    parse(*sys.argv[1:])
