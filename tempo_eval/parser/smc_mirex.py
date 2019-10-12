import csv
import sys
from os import walk
from os.path import join, basename, exists

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import timestamps_to_bpm, create_tempo_annotation, get_bibtex_entry, \
    create_jam
from tempo_eval.evaluation import get_references_path


def parse(*args, **kwargs):
    parse_holzapfel2012()
    parse_percival2014(args[0])


def parse_holzapfel2012():
    output_dir = get_references_path('smc_mirex', 'jams')
    smkdirs(output_dir)
    input_annotation_dir = get_references_path('smc_mirex', 'holzapfel2012')

    for (dirpath, _, filenames) in walk(input_annotation_dir):
        for filename in [f for f in filenames if f.endswith('.jams')]:
            jams_file = join(output_dir, filename)
            jam = jams.load(join(dirpath, filename))
            # add file identifier, because we did so for others
            jam.file_metadata.identifiers = {'file': filename.replace('.jams', '.wav')}

            # add missing version info
            jam.annotations['beat'][0].annotation_metadata.version = '1.0'
            jam.annotations['tag_open'][0].annotation_metadata.version = '1.0'
            # massage tag values, because we have '(ternary)' and 'ternary' -> remove parentheses
            tag_open = jam.annotations['tag_open'][0]
            original_observations = tag_open.pop_data()
            for o in original_observations:
                value = o.value
                if value.startswith('(') and value.endswith(')') or value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                # normalize tags, by packing tags with identical meaning together
                if value == 'ternary':
                    value = 'ternary meter'
                elif value == 'excerpt is the introduction of a song':
                    value = 'excerpt is the introduction of the song'
                elif value == 'pause':
                    value = 'pauses'
                elif value == 'tempo incontinuity':
                    value = 'tempo discontinuity'
                elif value == 'high syncopation':
                    value = 'strong syncopation'
                elif value == 'no repetition (melodic or rhythmic)':
                    value = 'no repetition'
                elif value == 'low familiarity':
                    value = 'low familiarity with the song/style'
                tag_open.append(time=o.time, duration=o.duration, value=value, confidence=o.confidence)

            # iterate over beats
            beats = jam.annotations['beat'][0]['data']
            timestamps = [b.time for b in beats]
            median_bpm, mean, std, cv = timestamps_to_bpm(timestamps)
            jam.annotations.append(_create_holzapfel2012_tempo_annotation(median_bpm))
            jam.save(jams_file)


def parse_percival2014(input_audio_dir):
    output_dir = get_references_path('smc_mirex', 'jams')
    input_annotation_file = get_references_path('smc_mirex', 'percival2014', 'smc_mirum_tempos.mf')
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
            jam.annotations.append(_create_percival2014_tempo_annotation(bpm))
            jam.save(jams_file)


def _create_percival2014_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus='SMC_MIREX',
        version='2.0',
        curator=jams.Curator(name='Graham Percival', email='graham@percival-music.ca'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='unknown',
        annotator={'bibtex': get_bibtex_entry('Percival2014'),
                   'ref_url': 'http://www.marsyas.info/tempo/'})
    return tempo


def _create_holzapfel2012_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus='SMC_MIREX',
        version='1.0',
        curator=jams.Curator(name='Matthew Davies', email='mdavies@inescporto.pt'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Holzapfel2012'),
                   'ref_url': 'https://repositorio.inesctec.pt/bitstream/123456789/2539/1/PS-07771.pdf'})
    return tempo


if __name__ == '__main__':
    parse(*sys.argv[1:])
