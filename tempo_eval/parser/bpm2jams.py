import argparse
import sys
import traceback
from os import walk
from os.path import join, exists

import jams

from tempo_eval.parser.util import create_tempo_annotation, create_jam, get_base, create_annotator
from tempo_eval.evaluation import get_estimates_path


def parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Converts files containing a single BPM value to jams.')

    parser.add_argument('-i', '--input', help='Input directory, relative to corpus, e.g. \'boeck2015/tempodetector2016_default\'')
    parser.add_argument('-w', '--input_audio', help='Input directory for audio files.')
    parser.add_argument('-c', '--corpus', help='Corpus name, e.g. \'ballroom\'')
    parser.add_argument('-a', '--annotation_tools', help='Annotations tools')
    parser.add_argument('-d', '--data_source', help='Data source')
    parser.add_argument('-v', '--version', help='Annotation version')
    parser.add_argument('-b', '--bibtex', help='Bibtex cite key')

    # parse arguments
    args = parser.parse_args()
    parse_bpm(args)


def parse_bpm(args):
    working_dir = get_estimates_path(args.corpus, args.input)
    if not exists(working_dir):
        print('Input directory does not exists: {}'.format(working_dir), file=sys.stderr)
    else:
        input_audio_dir = args.input_audio
        for (dirpath, _, filenames) in walk(working_dir):
            for filename in [f for f in filenames if not f.endswith('.jams') and not f.startswith('.')]:
                input_file = join(dirpath, filename)
                first_dot = input_file.find('.')
                if first_dot > 0:
                    jams_file = get_base(input_file) + '.jams'
                else:
                    jams_file = input_file + '.jams'
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        line = f.readline()
                        if line:
                            tempo = float(line)
                        else:
                            print('Empty file, assuming tempo 0: {}'.format(input_file), file=sys.stderr)
                            tempo = 0.
                        if exists(jams_file):
                            jam = jams.load(jams_file)
                        else:
                            jam = create_jam(filename, input_audio_dir)
                        jam.annotations.append(_create_single_tempo_annotation(tempo,
                                                                               corpus=args.corpus,
                                                                               annotation_tools=args.annotation_tools,
                                                                               version=args.version,
                                                                               data_source=args.data_source,
                                                                               bibtex=args.bibtex))
                        jam.save(jams_file)
                except Exception as e:
                    print('An error occurred when trying to convert {}: {} {}'.format(input_file, type(e).__name__, e),
                          file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)


def _create_single_tempo_annotation(tempo, corpus='', version='', annotation_tools='', data_source='', bibtex=None):
    tempo = create_tempo_annotation(tempo)
    annotator = create_annotator(bibtex)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=corpus,
        version=version,
        annotation_tools=annotation_tools,
        annotator=annotator,
        data_source=data_source)
    return tempo


if __name__ == '__main__':
    parse()
