"""
Parser for RWC data. See https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/

Beat format description:

Each line corresponds to a beat.
The first column is the start timing of the beat (in second).
The second column is the end timing of the beat (when we consider
each beat as a period having a duration, not as a point).
You could ignore this second column in usual as it is same with
the first column of the next line.

384 indicates the beginning of a bar (measure).
48 corresponds to a quarter-note beat (like MIDI tick = 48, which
was sometimes used in 1980s). If you consider 384 as 0, it means
the relative position from the beginning of each bar.

In the case of 3/4 time signature, the third column is going to be
"384, 48, 96, 384, 48, 96, 384, 48, 96, 384, 48, 96,...".

In the case of 4/4 time signature, the third column is going to be
"384, 48, 96, 144, 384, 48, 96, 144, 384, 48, 96, 144, ...".

In the case of 5/4 time signature, the third column is going to be
"384, 48, 96, 144, 192, 384, 48, 96, 144, 192, 384, 48, 96, 144,
192,...".

In the case of 6/8 time signature, the third column is going to be
"384, 24, 48, 72, 96, 120, 384, 24, 48, 72, 96, 120, 384, 24, 48,
72, 96, 120, ...".
"""

import logging
import math
import sys
from os import walk
from os.path import join, exists

import jams
import pandas as pd
import numpy as np
from jams.util import smkdirs

from tempo_eval.evaluation import get_references_path
from tempo_eval.parser.util import create_jam, timestamps_to_bpm, create_tempo_annotation, \
    create_beat_annotation, get_bibtex_entry, create_tag_open_annotation


def parse(*args, **kwargs):
    input_audio_dir = args[0]
    parse_goto(input_audio_dir, 'rwc_mdb_c', 'goto2006', _parse_classical_metadata)
    parse_goto(input_audio_dir, 'rwc_mdb_g', 'goto2006', _parse_genre_metadata)
    parse_goto(input_audio_dir, 'rwc_mdb_j', 'goto2006', _parse_jazz_metadata)
    parse_goto(input_audio_dir, 'rwc_mdb_p', 'goto2006', _parse_pop_metadata)
    parse_goto(input_audio_dir, 'rwc_mdb_r', 'goto2006', _parse_royalty_free_metadata)


def parse_goto(input_audio_dir, corpus, publication, metadata_parser):
    output_dir = get_references_path(corpus, 'jams')
    smkdirs(output_dir)
    input_annotation_dir = get_references_path(corpus, publication)

    copyright_note_file = join(get_references_path(corpus, publication), 'README.txt')

    # HTML formatted metadata from
    # https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/
    metadata_file = join(get_references_path(corpus, publication), 'metadata.html')

    metadata_df = None
    if exists(metadata_file):
        metadata_df = metadata_parser(metadata_file)

    with open(copyright_note_file, mode='r') as f:
        copyright_note = f.read()

    for (dirpath, _, filenames) in walk(input_annotation_dir):
        for filename in [f for f in filenames if f.endswith('.BEAT.TXT')]:
            id = filename.replace('.BEAT.TXT', '')
            jams_file = join(output_dir, filename.replace('.BEAT.TXT', '.jams'))
            if exists(jams_file):
                jam = jams.load(jams_file)
            else:
                artist = None
                title = None
                if metadata_df is not None:
                    artist = metadata_df.artist.at[id]
                    title = metadata_df.title.at[id]
                jam = create_jam(filename, input_audio_dir, artist=artist, title=title)
            with open(join(dirpath, filename), mode='r') as file:

                lines = file.readlines()

                meter, note_length = _extract_time_signature(lines)
                logging.debug('Signature: {}/{}'.format(meter, note_length))

                beats = []
                for line in lines:
                    splits = line.split()
                    timestamp = float(splits[0]) / 100.
                    ticks = int(splits[2])

                    beats.append({'timestamp': timestamp,
                                  'position': (_ticks_to_beat_position(ticks, meter, note_length)),
                                  'confidence': 1.0})
                timestamps = [b['timestamp'] for b in beats]
                median_bpm, _, _, cv = timestamps_to_bpm(timestamps, meter=meter)
                jam.annotations.append(_create_goto_tempo_annotation(median_bpm, corpus, copyright_note))
                jam.annotations.append(_create_goto_beat_annotation(beats, corpus, copyright_note))

                # if we have original tempo annotations, generate a ground truth for them.
                if metadata_df is not None and 'tempo' in metadata_df.columns:
                    tempo = float(metadata_df.tempo[id])
                    jam.annotations.append(
                        _create_original_goto_tempo_annotation(tempo, corpus, copyright_note)
                    )
                # since we know the signature, toss it in there as tag_open
                tags = ['{}/{}'.format(meter, note_length)]
                # if we have tags, add them
                if metadata_df is not None and 'tags' in metadata_df.columns:
                    tags.extend(metadata_df.tags[id])
                jam.annotations.append(_create_goto_tag_open_annotation(tags, corpus, copyright_note))
                jam.save(jams_file)


def _ticks_to_beat_position(ticks, meter, note_length):
    # for some reason, some entries are greater than 384.
    # E.g., in rwc_mdb_c/jams/RM-C025_D.BEAT.TXT
    while ticks > 384:
        ticks -= 384
    position = 0
    if ticks < 0:
        position = 0
    elif ticks != 384:
        if note_length == 2:
            position = ticks // 96 + 1
        elif note_length == 4:
            position = ticks // 48 + 1
        elif note_length == 8:
            position = ticks // 24 + 1
        else:
            logging.warning('Don\'t know what to do with '
                            'time signature {}/{}'
                            .format(meter, note_length))
    else:
        position = 1
    return position


def _piece_to_id(piece):
    if isinstance(piece, str):
        no_no = piece.replace('No. ', '')
        try:
            return '{:03d}'.format(int(no_no))
        except ValueError:
            return '0' + no_no
    else:
        return piece


def _parse_classical_metadata(metadata_file):
    tables = pd.read_html(metadata_file)
    df = tables[4]
    df.drop([0], inplace=True)
    # fix column names
    df.columns = ['id', 'catsuffix', 'tracknumber', 'title',
                  'composer', 'artist', 'length', 'tags']
    df.tracknumber = df\
        .tracknumber.transform(lambda x: int(x.replace('Tr. ', ''))
    if isinstance(x, str) else x)
    df.id = df.id\
        .transform(lambda x: 'RM-C' + _piece_to_id(x) if isinstance(x, str) else x)
    # fill in missing ids
    last_id = ''
    suite_rows = []
    letter = 'A'
    for i in range(len(df.id)):
        value = df.id.iat[i]
        if math.isnan(df.tracknumber.iat[i]):
            suite_rows.append(i)
        if not isinstance(value, str) and math.isnan(value):
            suite_row = suite_rows[-1]
            df.id.iat[i] = last_id + '_' + letter
            df.title.iat[i] = df.title.iat[suite_row] + ': ' + df.title.iat[i]
            df.catsuffix.iat[i] = df.catsuffix.iat[suite_row]
            df.composer.iat[i] = df.composer.iat[suite_row]
            df.artist.iat[i] = df.artist.iat[suite_row]
            df.tags.iat[i] = df.tags.iat[suite_row]
            letter = chr(ord(letter) + 1)
        else:
            last_id = value
            letter = 'A'
            # convert to list
            df.tags.iat[i] = [df.tags.iat[i]]
    # drop suites
    df.drop([df.index[s] for s in suite_rows], inplace=True)
    # use id as index
    df.set_index('id', inplace=True)
    return df


def _parse_genre_metadata(metadata_file):
    tables = pd.read_html(metadata_file)
    df = tables[4]
    df.drop([0], inplace=True)
    # fix column names
    df.columns = ['id', 'catsuffix', 'tracknumber', 'genre',
                  'subgenre', 'title', 'composer', 'artist', 'length']
    # add tags column
    df['tags'] = np.nan
    # make sure it has the correct type
    df.tags = df.tags.astype(list)

    df.tracknumber = df.tracknumber\
        .transform(lambda x: int(x.replace('Tr. ', '')) if isinstance(x, str) else x)
    df.id = df.id\
        .transform(lambda x: 'RM-G' + _piece_to_id(x) if isinstance(x, str) else x)
    for i in range(len(df.id)):
        # infer tags
        tags = list({df.genre.iat[i], df.subgenre.iat[i]})
        if ' (Male)' in df.artist.iat[i]:
            df.artist.iat[i] = df.artist.iat[i]\
                .replace(' (Male)', '')
            tags.append('male')
        elif ' (Female)' in df.artist.iat[i]:
            df.artist.iat[i] = df.artist.iat[i]\
                .replace(' (Female)', '')
            tags.append('female')
        elif '(2 Male & 2 Female)' in df.artist.iat[i]:
            tags.append('female')
            tags.append('male')
        df.tags.iat[i] = tags
    # use id as index
    df.set_index('id', inplace=True)
    return df


def _parse_jazz_metadata(metadata_file):
    tables = pd.read_html(metadata_file)
    df_tracks = tables[4]
    df_tracks.drop([0, 1], inplace=True)
    df_instruments = tables[5]

    # fix column names
    df_tracks.columns = ['id', 'catsuffix', 'tracknumber',
                         'title', 'artist', 'length', 'variation', 'instruments']
    df_tracks['composer'] = ''
    # add tags column
    df_tracks['tags'] = np.nan
    # make sure it has the correct type
    df_tracks.tags = df_tracks.tags.astype(list)

    df_tracks.tracknumber = df_tracks.tracknumber\
        .transform(lambda x: int(x.replace('Tr. ', '')) if isinstance(x, str) else x)
    df_tracks.id = df_tracks.id\
        .transform(lambda x: 'RM-J' + _piece_to_id(x) if isinstance(x, str) else x)

    # fix column names
    df_instruments.columns = ['id', 'name']
    df_instruments.id = df_instruments.id\
        .transform(lambda x: x.replace(':', ''))
    df_instruments.set_index('id', inplace=True)

    # fill in tags
    for i in range(len(df_tracks.id)):
        inst = df_tracks.instruments.iat[i]\
            .replace(' × 2', '').split(' & ')
        tags = [df_tracks.variation.iat[i]]
        tags.extend([df_instruments.name.at[i] for i in inst])
        df_tracks.tags.iat[i] = tags

    # use id as index
    df_tracks.set_index('id', inplace=True)
    return df_tracks


def _parse_pop_metadata(metadata_file):
    tables = pd.read_html(metadata_file)
    df_tracks = tables[4]
    df_tracks.drop([0], inplace=True)
    df_instruments = tables[5]

    # fix column names
    df_tracks.columns = ['id', 'catsuffix', 'tracknumber', 'title',
                         'artist', 'singer', 'length', 'tempo', 'instruments', 'drums']
    df_tracks['composer'] = ''
    # add tags column
    df_tracks['tags'] = np.nan
    # make sure it has the correct type
    df_tracks.tags = df_tracks.tags.astype(list)

    df_tracks.tracknumber = df_tracks.tracknumber\
        .transform(lambda x: int(x.replace('Tr. ', '')) if isinstance(x, str) else x)
    df_tracks.id = df_tracks.id\
        .transform(lambda x: 'RM-P' + _piece_to_id(x) if isinstance(x, str) else x)

    # fix column names
    df_instruments.columns = ['id', 'name']
    df_instruments.id = df_instruments.id.transform(lambda x: x.replace(':', ''))
    df_instruments.set_index('id', inplace=True)

    # fill in tags
    for i in range(len(df_tracks.id)):
        singer = df_tracks.singer.iat[i]
        tags = [singer.replace(' [English]', ''), df_tracks.drums.iat[i]]
        if ' [English]' in singer:
            tags.append('English vocals')
        if isinstance(df_tracks.instruments.iat[i], str):
            inst = df_tracks.instruments.iat[i].replace(' × 2', '').split(' & ')
            tags.extend([df_instruments.name.at[i] for i in inst])
        df_tracks.tags.iat[i] = tags

    # use id as index
    df_tracks.set_index('id', inplace=True)
    return df_tracks


def _parse_royalty_free_metadata(metadata_file):
    tables = pd.read_html(metadata_file)
    df_tracks = tables[4]
    df_tracks.drop([0], inplace=True)
    df_instruments = tables[5]

    # fix column names
    df_tracks.columns = ['id', 'catsuffix', 'tracknumber', 'title',
                         'artist', 'singer', 'length', 'tempo', 'instruments', 'drums']
    df_tracks['composer'] = ''
    # add tags column
    df_tracks['tags'] = np.nan
    # make sure it has the correct type
    df_tracks.tags = df_tracks.tags.astype(list)

    df_tracks.tracknumber = df_tracks.tracknumber\
        .transform(lambda x: int(x.replace('Tr. ', '')) if isinstance(x, str) else x)
    df_tracks.id = df_tracks.id\
        .transform(lambda x: 'RM-R' + _piece_to_id(x) if isinstance(x, str) else x)

    # fix column names
    df_instruments.columns = ['id', 'name']
    df_instruments.id = df_instruments.id.transform(lambda x: x.replace(':', ''))
    df_instruments.set_index('id', inplace=True)

    # fill in tags
    for i in range(len(df_tracks.id)):
        singer = df_tracks.singer.iat[i]
        tags = [singer.replace(' [English]', ''), df_tracks.drums.iat[i]]
        if ' [English]' in singer:
            tags.append('English vocals')
        if isinstance(df_tracks.instruments.iat[i], str):
            inst = df_tracks.instruments.iat[i].replace(' × 2', '').split(' & ')
            tags.extend([df_instruments.name.at[i] for i in inst])
        df_tracks.tags.iat[i] = tags

    # use id as index
    df_tracks.set_index('id', inplace=True)
    return df_tracks


def _extract_time_signature(lines):

    denominator = 1
    max_ticks = 0
    min_ticks = 384

    for line in lines:
        splits = line.split()
        ticks = int(splits[2])

        if ticks < 0:
            continue

        # for some reason, some entries are greater than 384.
        # E.g., in rwc_mdb_c/jams/RM-C025_D.BEAT.TXT
        while ticks > 384:
            ticks -= 384

        if ticks != 384:
            max_ticks = max(max_ticks, ticks)
            min_ticks = min(min_ticks, ticks)

    if min_ticks == 12:
        denominator = 16
    elif min_ticks == 24:
        denominator = 8
    elif min_ticks == 48 or min_ticks == 47:  # weird exception for RM-P064
        denominator = 4
    elif min_ticks == 96:
        denominator = 2

    numerator = (max_ticks // min_ticks) + 1
    return numerator, denominator


def _create_goto_tempo_annotation(bpm, corpus, copyright_note):
    tempo = create_tempo_annotation(bpm, license=copyright_note)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=corpus,
        version='1.0',
        curator=jams.Curator(name='Masataka Goto', email='m.goto@aist.go.jp'),
        data_source='manual annotation',
        annotation_tools='derived from beat annotations',
        annotation_rules='median of corresponding inter beat intervals',
        annotator={
            'name': 'Masataka Goto',
            'bibtex': get_bibtex_entry('Goto2006'),
            'ref_url': 'https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/'
        }
    )
    return tempo


def _create_original_goto_tempo_annotation(bpm, corpus, copyright_note):
    tempo = create_tempo_annotation(bpm, license=copyright_note)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=corpus,
        version='0.1',
        curator=jams.Curator(name='Masataka Goto', email='m.goto@aist.go.jp'),
        data_source='AIST website. Tempo values are rough '
                    'estimates and should not be used as data for research purposes.',
        annotation_tools='unknown',
        annotation_rules='unknown',
        annotator={
            'name': 'Masataka Goto',
            'bibtex': get_bibtex_entry('Goto2006'),
            'ref_url': 'https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/'
        }
    )
    return tempo


def _create_goto_beat_annotation(beats, corpus, copyright_note):
    beat = create_beat_annotation(beats, license=copyright_note)
    beat.annotation_metadata = jams.AnnotationMetadata(
        corpus=corpus,
        version='1.0',
        curator=jams.Curator(name='Masataka Goto', email='m.goto@aist.go.jp'),
        data_source='manual annotation',
        annotator={
            'name': 'Masataka Goto',
            'bibtex': get_bibtex_entry('Goto2006'),
            'ref_url': 'https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/'
        }
    )
    return beat


def _create_goto_tag_open_annotation(tags, corpus, copyright_note):
    tag = create_tag_open_annotation(*tags, license=copyright_note)
    tag.annotation_metadata = jams.AnnotationMetadata(
        corpus=corpus,
        version='1.0',
        curator=jams.Curator(name='Masataka Goto', email='m.goto@aist.go.jp'),
        data_source='manual annotation',
        annotator={
            'name': 'Masataka Goto',
            'bibtex': get_bibtex_entry('Goto2006'),
            'ref_url': 'https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/'
        }
    )
    return tag


if __name__ == '__main__':
    parse(*sys.argv[1:])