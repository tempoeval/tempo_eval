import logging
import sys
from os import walk
from os.path import join, exists

import jams
from jams.util import smkdirs

from tempo_eval.evaluation import get_references_path
from tempo_eval.parser.util import create_jam, timestamps_to_bpm, create_tempo_annotation, \
    get_bibtex_entry, create_beat_annotation

HJDB = 'hjdb'


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_boeck2019(input_audio_dir)


def parse_boeck2019(input_audio_dir):
    output_dir = get_references_path('hjdb', 'jams')
    smkdirs(output_dir)
    input_annotation_dir = get_references_path('hjdb', 'boeck2019')

    for (dirpath, _, filenames) in walk(input_annotation_dir):
        for filename in [f for f in filenames if f.endswith('.beats')]:
            jams_file = join(output_dir, filename.replace('.beats', '.jams'))
            if exists(jams_file):
                logging.warning('Adding to an existing jams file: {}, {}'.format(jams_file, filename))
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename.replace('.beats', '.wav'), input_audio_dir,
                                 title=filename.replace('.beats', '').replace('_', ' '))

            # we assume these derived BPM values are by S. Böck.
            bpm_file = filename.replace('.beats', '.bpm')
            with open(join(dirpath, bpm_file), mode='r') as file:
                line = file.readline()
                if line:
                    tempo = float(line)
                else:
                    print('Empty file, assuming tempo 0: {}'.format(bpm_file), file=sys.stderr)
                    tempo = 0.
                jam.annotations.append(_create_boeck2019_tempo_annotation(tempo))

            # we assume the beat annotations are by J. Hockman
            with open(join(dirpath, filename), mode='r') as file:
                lines = file.readlines()
                beats = []
                for line in lines:
                    splits = line.split()
                    beats.append({'timestamp': float(splits[0]), 'position': int(splits[1]), 'confidence': 1.0})
                meter = max([b['position'] for b in beats])
                timestamps = [b['timestamp'] for b in beats]

                # ICBI-based
                median_bpm, _, _, _ = timestamps_to_bpm(timestamps, meter=meter)
                jam.annotations.append(_create_hockman2012_cor_beat_tempo_annotation(median_bpm))

                # IBI-based
                median_bpm, _, _, _ = timestamps_to_bpm(timestamps)
                jam.annotations.append(_create_hockman2012_tempo_annotation(median_bpm))

                jam.annotations.append(_create_hockman2012_beat_annotation(beats))
                jam.save(jams_file)


def _create_boeck2019_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=HJDB,
        version='3.0',
        curator=jams.Curator(name='Sebastian Böck', email='sebastian.boeck@ofai.at'),
        data_source='GitHub repository of Sebastian Böck',
        annotation_tools='unknown',
        annotation_rules='unknown',
        annotator={'bibtex': get_bibtex_entry('Boeck2019'),
                   'ref_url': 'https://github.com/superbock/ISMIR2019'})
    return tempo


def _create_hockman2012_cor_beat_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=HJDB,
        version='2.0',
        curator=jams.Curator(name='Jason Hockman', email='Jason.Hockman@bcu.ac.uk'),
        data_source='manual annotation, GitHub repository of Sebastian Böck',
        annotation_tools='derived from beat annotations',
        annotation_rules='based on median of inter corresponding beat intervals',
        annotator={'bibtex': get_bibtex_entry('Hockman2012'),
                   'ref_url': 'https://github.com/superbock/ISMIR2019'})
    return tempo


def _create_hockman2012_tempo_annotation(bpm):
    tempo = create_tempo_annotation(bpm)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=HJDB,
        version='1.0',
        curator=jams.Curator(name='Jason Hockman', email='Jason.Hockman@bcu.ac.uk'),
        data_source='manual annotation, GitHub repository of Sebastian Böck',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of inter beat intervals',
        annotator={'bibtex': get_bibtex_entry('Hockman2012'),
                   'ref_url': 'https://github.com/superbock/ISMIR2019'})
    return tempo


def _create_hockman2012_beat_annotation(beats):
    beat = create_beat_annotation(beats)
    beat.annotation_metadata = jams.AnnotationMetadata(
        corpus=HJDB,
        version='1.0',
        curator=jams.Curator(name='Jason Hockman', email='Jason.Hockman@bcu.ac.uk'),
        data_source='manual annotation, GitHub repository of Sebastian Böck',
        annotator={'bibtex': get_bibtex_entry('Hockman2012'),
                   'ref_url': 'https://github.com/superbock/ISMIR2019'})
    return beat


if __name__ == '__main__':
    parse(*sys.argv[1:])
