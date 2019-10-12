import argparse
import sys
import traceback
from os import walk
from os.path import join, exists

import jams

from tempo_eval.parser.util import create_tempo_annotation, create_jam, get_bibtex_entry, get_base, create_annotator
from tempo_eval.evaluation import get_estimates_path


def parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Converts mirex style tempo estimation (T1 T2 ST1, T1<T2) to jams.')

    parser.add_argument('-i', '--input', help='Input directory, relative to corpus, e.g. \'boeck2015/tempodetector2016_default\'')
    parser.add_argument('-w', '--input_audio', help='Input directory for audio files.')
    parser.add_argument('-c', '--corpus', help='Corpus name, e.g. \'ballroom\'')
    parser.add_argument('-a', '--annotation_tools', help='Annotations tools')
    parser.add_argument('-v', '--version', help='Annotation version')
    parser.add_argument('-b', '--bibtex', help='Bibtex cite key(s)', nargs='*')

    # parse arguments
    args = parser.parse_args()
    parse_mirex(args)


def parse_mirex(args):
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
                        splits = line.split()
                        T1 = float(splits[0])
                        T2 = float(splits[1])
                        ST1 = float(splits[2])
                        if T2 < T1:
                            print('Warning: T2({}) is less than T1({}).'.format(T2, T1))
                        if exists(jams_file):
                            jam = jams.load(jams_file)
                        else:
                            jam = create_jam(filename, input_audio_dir)
                        jam.annotations.append(_create_mirex_tempo_annotation(T1, T2, ST1,
                                                                              corpus=args.corpus,
                                                                              annotation_tools=args.annotation_tools,
                                                                              version=args.version,
                                                                              bibtex=args.bibtex))
                        jam.save(jams_file)
                except Exception as e:
                    print('An error occurred when trying to convert {}: {} {}'.format(input_file, type(e).__name__, e),
                          file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)


def _create_mirex_tempo_annotation(t1, t2, st1, corpus='', version='', annotation_tools='', bibtex=None):
    tempo = create_tempo_annotation(t1, tempo2=t2, confidence1=st1)
    annotator = create_annotator(bibtex)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=corpus,
        version=version,
        annotation_tools=annotation_tools,
        annotator=annotator)
    return tempo


if __name__ == '__main__':
    parse()
