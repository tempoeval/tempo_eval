"""
Command line scripts.
"""
import argparse
import csv
import json
import logging
from os import walk
from os.path import exists, isfile, dirname, splitext, join, basename

import audioread
import jams
from jams.util import smkdirs

from tempo_eval import version
from tempo_eval.report import Size, print_report
from tempo_eval.version import version


def tempo_eval_command():
    """tempo_eval"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
The program 'tempo_eval' creates Markdown formatted tempo estimation
evaluation reports.

To provide multiple estimates/references for a single track, place
multiple annotations in the same jam file, but make sure to use a
different annotation metadata version. Alternatively, use multiple
jam files with the same name placed into different folders, again
using different annotations metadata versions.

Unless otherwise specified, tempo_eval creates reports for all known corpora.
However, if estimates or references are explicitly specified, only
reports for corpora occurring in the user-provided annotations are generated. 

License: ISC License
''')

    parser.add_argument('-v', '--version',
                        action='version',
                        version=version)
    parser.add_argument('-e', '--estimates',
                        help='Folder structure with estimate jams. Will be parsed recursively. '
                             'Estimates are distinguished based on their jam file name and '
                             'their annotation metadata version.',
                        required=False)
    parser.add_argument('-r', '--references',
                        help='Folder structure with reference jams.  Will be parsed recursively. '
                             'References are distinguished based on their jam file name and '
                             'their annotation metadata version.',
                        required=False)
    parser.add_argument('-c', '--corpus',
                        nargs='*',
                        help='List of corpora to generate reports for.',
                        required=False)
    parser.add_argument('-d', '--dir',
                        help='Output directory.',
                        default='.',
                        required=False)
    parser.add_argument('--validate',
                        action='store_true',
                        help='Validate jams when reading. May cause very slow execution.',
                        default=False,
                        required=False)
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        help='Verbose output during generation.',
                        default=False,
                        required=False)
    parser.add_argument('-f', '--format',
                        const='kramdown',
                        default='kramdown',
                        nargs='?',
                        choices=['html', 'kramdown', 'markdown'],
                        help='Output format. Choose \'kramdown\' for GitHub Pages.',
                        required=False)
    parser.add_argument('-s', '--size',
                        const='L',
                        default='L',
                        nargs='?',
                        choices=['S', 'M', 'L', 'XL'],
                        help='Size of evaluation report. Choose \'L\' for the '
                             'comprehensive version, choose \'S\' for a '
                             'shorter version with just the essentials. \'XL\' may '
                             'contain experimental metrics.',
                        required=False)

    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
        log_format = '%(asctime)s %(levelname)-8s %(message)s'
    else:
        level = logging.WARNING
        log_format = '%(levelname)-8s %(message)s'

    logging.getLogger().setLevel(level)
    logging.basicConfig(format=log_format)
    logging.debug('Log level: {}'.format(level))
    logging.debug('Report size: {}'.format(args.size))

    print_report(output_dir=args.dir,
                 validate=args.validate,
                 corpus_names=args.corpus,
                 estimates_dir=args.estimates,
                 references_dir=args.references,
                 format=args.format,
                 size=Size[args.size])


def convert2jams_command():
    """convert2jams"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
The program 'convert2jams' converts tempo annotations from different formats
to JAMS.

Supported are:

 - Simple text files containing either a single BPM value or MIREX-style
   values, i.e. "t1 t2 s1" separated by whitespace. The name of such a
   textfile minus its extension is used as name for the jam file.
 - JSON files consisting of a single dictionary with names as keys and
   either a single float as value or a list of three values (MIREX).
   Names must not contain a file extension.
 - CSV/TSV files containing two columns, name and value. Names must not
   contain a file extension.   

Since JAMS require a duration, this script makes an effort to find a
corresponding audio file and read its duration. To achieve this, it makes
the assumption that the audio file name corresponds to the name of either
the used annotation file or the annotation name. E.g. an annotation file
"my_track.txt" should correspond to an audio file named "my_track.mp3"
or "my_track.wav" (any format that can be read by the library 'audioread' is
supported). Note that annotation names should be unique. 

License: ISC License
''')

    parser.add_argument('-d', '--dir',
                        help='Output directory.',
                        required=False)
    parser.add_argument('-i', '--input',
                        help='Textual input file or directory '
                             'containing your annotations.',
                        required=True)
    parser.add_argument('-a', '--input-audio',
                        help='Input directory for audio '
                             'files (required to derived duration).',
                        required=True)
    parser.add_argument('-c', '--corpus',
                        help='Corpus name, e.g. \'ballroom\'. '
                             'Required unless a template is used.')
    parser.add_argument('-v', '--annotation-version',
                        help='Annotation version. '
                             'Required unless a template is used')
    parser.add_argument('-t', '--annotation-tools',
                        help='Annotations tools',
                        default='')
    parser.add_argument('-r', '--annotation-rules',
                        help='Annotations rules',
                        default='')
    parser.add_argument('-w', '--validation',
                        help='Validation',
                        default='')
    parser.add_argument('-s', '--data-source',
                        help='Data source',
                        default='')
    parser.add_argument('-n', '--curator-name',
                        help='Curator name',
                        default='')
    parser.add_argument('-e', '--curator-email',
                        help='Curator email',
                        default='')
    parser.add_argument('-b', '--bibtex',
                        help='Bibtex record. May either be '
                             'a file (which will be read) or a string',
                        default='')
    parser.add_argument('-u', '--ref-url',
                        help='Reference URL',
                        default='')
    parser.add_argument('-f', '--flatten',
                        help='Flatten directory structure. '
                             'Only supported for input files.',
                        action='store_true')
    parser.add_argument('--template',
                        help='Template JAMS file containing a '
                             'single annotation metadata block, '
                             'to be used as template.')
    parser.add_argument('--version',
                        action='version',
                        version=version)
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        help='Verbose output during generation.',
                        default=False,
                        required=False)

    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
        log_format = '%(asctime)s %(levelname)-8s %(message)s'
    else:
        level = logging.WARNING
        log_format = '%(levelname)-8s %(message)s'

    logging.getLogger().setLevel(level)
    logging.basicConfig(format=log_format)
    logging.debug('Log level: {}'.format(level))

    if args.template:
        logging.debug('Using provided annotation metadata template: {}'
                     .format(args.template))
        annotation_metadata = _read_annotation_metadata_template(args.template,
                                                                 args.verbose)
        if isinstance(annotation_metadata, int):
            return annotation_metadata
    else:
        logging.debug('Using annotation metadata provided on as parameters.')

        if not args.corpus:
            logging.fatal('No corpus name provided.')
            return 2

        if not args.annotation_version:
            logging.fatal('No annotation version provided.')
            return 2

        bibtex = args.bibtex
        if exists(bibtex) and isfile(bibtex):
            logging.info('Reading bibtex from file {} ...'.format(bibtex))
            with open(bibtex, 'r', encoding='utf-8') as f:
                bibtex = f.read()

        annotation_metadata = jams.AnnotationMetadata(
            corpus=args.corpus,
            version=args.annotation_version,
            annotation_tools=args.annotation_tools,
            data_source=args.data_source,
            annotation_rules=args.annotation_rules,
            validation=args.validation,
            curator={'name': args.curator_name,
                     'email': args.curator_email},
            annotator={'bibtex': bibtex,
                       'ref_url': args.ref_url},
        )

    if not exists(args.input):
        logging.fatal('Input file/dir does not exists: {}'
                      .format(args.input))
        return 2

    durations = _read_audio_durations(args.input_audio)

    if isfile(args.input):

        if args.dir:
            out_dir = args.dir
            smkdirs(out_dir)
        else:
            out_dir = dirname(args.input)

        return _create_jams_from_file(args.input, out_dir,
                                      annotation_metadata,
                                      durations, args.flatten)
    else:
        if args.flatten:
            logging.fatal('Flatten is currently only supported '
                          'for input files, not dirs.')
            return 2

        if args.dir:
            out_dir = args.dir
            smkdirs(out_dir)
        else:
            out_dir = args.input

        return _create_jams_from_dir(args.input, out_dir,
                                     annotation_metadata,
                                     durations)


def _read_audio_durations(audio_dir):
    logging.info('Reading durations of your audio files from {} ...'
                 .format(audio_dir))
    durations = {}
    for (dirpath, _, filenames) in walk(audio_dir):
        for filename in filenames:

            splits = splitext(filename)
            name = splits[0]
            ext = splits[1].lower()

            # skip a bunch of files right away to avoid IO ops
            if ext in ['.txt', '.csv', '.tsv', '.mirex', '.bpm', '.mf', '.jams'] \
                    or filename.startswith('.'):
                continue

            audio_file_name = join(dirpath, filename)
            try:
                with audioread.audio_open(audio_file_name) as file:
                    if name in durations:
                        logging.warning('Audio file name is not unique: {}'.format(name))
                    else:
                        durations[name] = filename, file.duration
            except:
                # fail (more or less) silently, as some files
                # may not even be audio files
                logging.debug('Failed to extract audio duration: {}'
                              .format(audio_file_name))

    return durations


def _create_jams_from_dir(input_dir, out_dir, metadata, durations):
    logging.info('Creating JAMS in {} ...'.format(out_dir))
    for (dirpath, _, filenames) in walk(input_dir):
        for filename in [f for f in filenames
                         if not f.endswith('.jams') and not f.startswith('.')]:
            input_file = join(dirpath, filename)
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                try:
                    # assume single tempo
                    tempo = float(content)

                except:
                    try:
                        # assume mirex style tempo
                        splits = content.split()
                        tempo = (float(splits[0]),
                                 float(splits[1]),
                                 float(splits[2]))
                        if tempo[0] > tempo[1]:
                            logging.warning('T2({t2}) is less than T1({t1}): {input}'
                                            .format(t1=tempo[0],
                                                    t2=tempo[1],
                                                    input=input_file))
                    except:
                        logging.error('Failure to parse. Use either a single '
                                      'tempo or MIREX style formatting: {}'
                                      .format(input_file))
                        continue

                name = splitext(basename(input_file))[0]

                # deal with .bpm.txt extensions
                second_split = splitext(name)
                if second_split[1].lower() == '.bpm':
                    name = second_split[0]

                ret = _create_jam(out_dir, durations,
                                  name, tempo, metadata)
                if ret:
                    return ret


def _create_jams_from_file(input_file, output_dir, metadata, durations, flatten):
    logging.info('Creating JAMS in {} ...'.format(output_dir))
    with open(input_file, mode='r') as file:

        if input_file.lower().endswith('.json'):
            d = json.load(file)
            for name, tempo in d.items():

                if flatten:
                    name = basename(name)
                ret = _create_jam(output_dir, durations, name, tempo, metadata)
                if ret:
                    return ret

        else:
            if input_file.lower().endswith('.csv'):
                reader = csv.reader(file, delimiter=',')
            elif input_file.lower().endswith('.tsv'):
                reader = csv.reader(file, delimiter='\t')
            else:
                logging.fatal('Input data format is not supported. '
                              'Please try CSV, TSV or JSON.')
                return 2

            for row in reader:

                try:
                    name = row[0]
                    if flatten:
                        name = basename(row[0])
                    tempo = float(row[1])
                    ret = _create_jam(output_dir, durations, name, tempo, metadata)
                    if ret:
                        return ret

                except Exception as e:
                    logging.error('Failed to create jam for row ({}): {}'
                                  .format(str(e), row))


def _create_jam(out_dir, durations, name, tempo, annotation_metadata):
    jams_file = join(out_dir, name + '.jams')
    logging.debug('Creating jam for {name} and {tempo}: {jam}'.format(name=name, tempo=tempo, jam=jams_file))

    if splitext(name)[1]:
        logging.warning('Name contains extension: {}'.format(name))

    if exists(jams_file):
        logging.error('JAMS file already exists: {}'.format(jams_file))
        return 2

    jam = jams.JAMS()
    annotation = jams.Annotation(namespace='tempo')
    if isinstance(tempo, float):
        annotation.append(time=0.0,
                          duration='nan',
                          value=tempo,
                          confidence=1.0)
    else:
        try:
            if len(tempo) == 3:
                annotation.append(time=0.0,
                                  duration='nan',
                                  value=tempo[0],
                                  confidence=tempo[2])
                annotation.append(time=0.0,
                                  duration='nan',
                                  value=tempo[1],
                                  confidence=1.-tempo[2])
            else:
                logging.error('Ignoring tempo annotations. '
                              'Unknown format ({}): {}'
                              .format(tempo, name))
                return
        except:
            logging.error('Ignoring tempo annotations. '
                          'Unknown format ({}): {}'
                          .format(tempo, name))
            return 0

    annotation.annotation_metadata = annotation_metadata
    jam.annotations.append(annotation)
    duration_key = basename(name)
    if duration_key in durations:
        audio_file_name, duration = durations[duration_key]
        jam.file_metadata.duration = duration
        jam.file_metadata.identifiers = {'file': audio_file_name}
    else:
        logging.warning('Audio duration not found. '
                        'Using dummy value (9999): {}'.format(name))
        jam.file_metadata.duration = 9999.

    jam.save(jams_file)


def _read_annotation_metadata_template(template, verbose):
    if not exists(template):
        logging.fatal('Annotation metadata template file does not exists: {}'
                      .format(template))
        return 2

    if not isfile(template):
        logging.fatal('Annotation metadata template is not a file: {}'
                      .format(template))
        return 2

    try:
        metadata = jams.load(template).annotations[0].annotation_metadata

        if not metadata.version:
            logging.fatal('Template annotation metadata must contain version.')
            return 2

        if not metadata.corpus:
            logging.fatal('Template annotation metadata must contain corpus name.')
            return 2

        return metadata

    except Exception as e:
        logging.fatal('Failed to extract annotation metadata '
                      'from template file: {}\nError: {}'
                      .format(template, str(e)))
        if verbose:
            raise e
        else:
            return 2


if __name__ == '__main__':
    tempo_eval_command()

