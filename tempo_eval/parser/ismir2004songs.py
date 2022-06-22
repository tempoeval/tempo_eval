import csv
import sys
from os.path import join, exists, dirname

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, create_tempo_annotation, get_bibtex_entry
from tempo_eval.evaluation import get_references_path


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_ismir2004(input_audio_dir)


def parse_ismir2004(input_audio_dir):
    output_dir = get_references_path('ismir2004songs', 'jams')
    input_annotation_file = get_references_path('ismir2004songs', 'ismir2004', 'ismir2004_songs_tempos.mf')
    with open(input_annotation_file, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            artist, release, title = _artist_release_title(row[0])
            # replicate despicable dir structure to avoid duplicate file names
            # like '01-AudioTrack 01.wav'
            filename = row[0].replace('MARSYAS_DATADIR/ismir2004_songs/', '')
            bpm = float(row[1])
            jams_file = join(output_dir, filename.replace('.wav', '.jams'))
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, input_audio_dir, artist=artist, title=title, release=release)
            jam.annotations.append(_create_ismir2004_annotation(bpm))
            smkdirs(dirname(jams_file))
            jam.save(jams_file)


def _artist_release_title(file_name):
    file_name = file_name.replace('/20sec/', '/')\
        .replace('/20-sec/', '/')\
        .replace('/20 sec/', '/')\
        .replace('_', ' ')\
        .replace('.wav', '')
    splits = file_name.split('/')
    parts = len(splits)
    pos = 1
    if parts < 5:
        # lets not make this too complicated
        # most of this is guessing anyway
        title = splits[parts-pos]
        release = ''
        artist = ''
    else:
        title = splits[parts-pos]
        pos += 1
        release = splits[parts-pos]
        pos += 1
        artist = splits[parts-pos]
    return artist, release, title


def _create_ismir2004_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus='ismir2004songs',
        version='1.0',
        curator=jams.Curator(name='Fabien Gouyon', email='fgouyon@pandora.com'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Gouyon2006'),
                   'ref_url': 'http://mtg.upf.edu//ismir2004_auco/contest/tempoContest/node6.html'})
    return tempo


if __name__ == '__main__':
    parse(*sys.argv[1:])
