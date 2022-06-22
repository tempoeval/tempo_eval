import argparse
import csv
import sys
from os.path import join, exists, dirname, basename

import jams
from jams.util import smkdirs

from tempo_eval.parser.util import create_tempo_annotation, create_jam, create_annotator
from tempo_eval.evaluation import get_estimates_path


def parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Converts .mf files containing multiple BPM values (one per audio file) to jams.')

    parser.add_argument('-i', '--input', help='Input .mf file, relative to corpus, e.g. \'gkiokas2012/default/estimates.mf\'')
    parser.add_argument('-w', '--input_audio', help='Input directory for audio files.')
    parser.add_argument('-c', '--corpus', help='Corpus name, e.g. \'ballroom\'')
    parser.add_argument('-a', '--annotation_tools', help='Annotations tools')
    parser.add_argument('-d', '--data_source', help='Data source')
    parser.add_argument('-v', '--version', help='Annotation version')
    parser.add_argument('-b', '--bibtex', help='Bibtex cite key')
    parser.add_argument('-f', '--flatten', help='Flatten directory structure', action='store_true')

    # parse arguments
    args = parser.parse_args()
    parse_mf(args)


def parse_mf(args):
    input_annotation_file = get_estimates_path(args.corpus, dirname(args.input), basename(args.input))
    if not exists(input_annotation_file):
        print('Input file does not exists: {}'.format(input_annotation_file), file=sys.stderr)
    else:
        input_audio_dir = args.input_audio
        output_dir = dirname(input_annotation_file)

        with open(input_annotation_file, mode='r') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if args.flatten:
                    filename = basename(row[0])
                else:
                    filename = row[0]  # use relative file names
                tempo = float(row[1])
                jams_file = join(output_dir, filename.replace('.wav', '.jams'))
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
                smkdirs(dirname(jams_file))
                jam.save(jams_file)


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
