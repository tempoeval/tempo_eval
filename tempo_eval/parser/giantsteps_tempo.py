import json
from os import walk
from os.path import join, exists

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, create_tempo_annotation, get_bibtex_entry
from tempo_eval.evaluation import get_references_path

GIANT_STEPS_TEMPO = 'GiantSteps Tempo'


def parse(*args, **kwargs):
    parse_knees2015()
    parse_schreiber2018()


def parse_knees2015():
    output_dir = get_references_path('giantsteps_tempo', 'jams')
    smkdirs(output_dir)
    input_annotation_dir = get_references_path('giantsteps_tempo', 'knees2015')

    for (dirpath, _, filenames) in walk(input_annotation_dir):
        for filename in [f for f in filenames if f.endswith('.jams')]:
            jams_file = join(output_dir, filename)
            jam = jams.load(join(dirpath, filename))
            # add file identifier, because we did so for others
            jam.file_metadata.identifiers = {'file': filename.replace('.jams', '.mp3')}
            jam.save(jams_file)


def parse_schreiber2018():
    output_dir = get_references_path('giantsteps_tempo', 'jams')
    input_annotation_file = get_references_path('giantsteps_tempo', 'schreiber2018', 'gs_new.json')

    with open(input_annotation_file, mode='r', encoding='utf-8') as file:
        data = json.load(file)['data']
        for beatport_id, values in data.items():
            filename = beatport_id + '.LOFI.jams'
            if values['no_beat']:
                continue
            mirex = values['mirex']
            bpm1 = mirex['T1']
            bpm2 = mirex['T2']
            salience1 = mirex['ST1']
            jams_file = join(output_dir, filename)
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                jam = create_jam(filename, None)
            jam.annotations.append(_create_schreiber2017_tempo_annotation(bpm1, bpm2, salience1))
            jam.save(jams_file)


def _create_schreiber2017_tempo_annotation(bpm1, bpm2, salience1):
    tempo = create_tempo_annotation(bpm1, bpm2, salience1)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=GIANT_STEPS_TEMPO,
        version='2.0',
        curator=jams.Curator(name='Hendrik Schreiber', email='hs@tagtraum.com'),
        data_source='crowdsource',
        annotation_tools='crowdsourced, web-based experiment',
        annotator={'bibtex': get_bibtex_entry('Schreiber2018b'),
                   'ref_url': 'http://www.tagtraum.com/tempo_estimation.html'})
    return tempo


if __name__ == '__main__':
    parse()
