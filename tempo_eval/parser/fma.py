import argparse
import ast
import os
import sys
from os.path import join, exists, dirname

import audioread
import jams
import pandas as pd
from jams.util import smkdirs

from tempo_eval.parser.util import create_jam, get_bibtex_entry, \
    create_tag_fma_genre_annotation
from tempo_eval.evaluation import get_references_path


def parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Imports FMA data to jams.')

    parser.add_argument('-f', '--fma', help='Path to extracted fma_metadata')
    parser.add_argument('-w', '--input_audio', help='Input directory for audio files.')
    parser.add_argument('-s', '--subset', help='small, medium, or large', default='small')
    parser.add_argument('-v', '--version', help='Annotation version', default='1.0.0')
    parser.add_argument('-b', '--bibtex', help='Bibtex cite key', default='Defferrard2017')
    parser.add_argument('-a', '--annotation_tools', help='Annotations tools', default='')
    parser.add_argument('-d', '--data_source', help='Data source', default='Free Music Archive, https://freemusicarchive.org/, https://github.com/mdeff/fma')

    # parse arguments
    args = parser.parse_args()

    tracks = _load(join(args.fma, 'tracks.csv'))
    small_tracks_df =  tracks.loc[tracks['set', 'subset'] == args.subset]
    corpus = 'fma_' + args.subset
    jams_dir = get_references_path(corpus, 'jams')
    input_audio_dir = args.input_audio

    for track_id, row in small_tracks_df.iterrows():
        artist = row['artist', 'name']
        album = row['album', 'title']
        title = row['track', 'title']
        genre = row['track', 'genre_top']
        # read real duration of sample, not whole track
        duration = _read_duration(track_id, input_audio_dir)

        jams_file = join(jams_dir, '{:06d}'.format(track_id)[0:3], '{:06d}.jams'.format(track_id))
        smkdirs(dirname(jams_file))
        if exists(jams_file):
            jam = jams.load(jams_file)
        else:
            jam = create_jam(jams_file, input_audio_dir, artist=artist, release=album, title=title, duration=duration)
        jam.file_metadata.identifiers = {'track_id': track_id}
        jam.annotations.append(_create_genre_annotation(genre,
                                                        corpus=corpus,
                                                        annotation_tools=args.annotation_tools,
                                                        version=args.version,
                                                        data_source=args.data_source,
                                                        bibtex=args.bibtex))
        jam.save(jams_file)


def _read_duration(track_id, input_audio_dir):
    audio_file = join(input_audio_dir, '{:06d}'.format(track_id)[0:3], '{:06d}.mp3'.format(track_id))
    duration = 30.
    try:
        with audioread.audio_open(audio_file) as file:
            duration = file.duration
    except:
        print('Failed to read audio file {}'.format(audio_file), file=sys.stderr)
    return duration


def _load(filepath):

    filename = os.path.basename(filepath)

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def _create_genre_annotation(genre, corpus='', version='', annotation_tools='', data_source='', bibtex=None):
    tag = create_tag_fma_genre_annotation(genre)
    annotator = {}
    if bibtex:
        entry = get_bibtex_entry(bibtex)
        if entry:
            annotator = {'bibtex': entry}
        else:
            print('Failed to find bibtex entry for cite-key \'{}\'. '
                  'Please make sure it occurs in file tempo_eval/references.bib.'.format(bibtex))
    tag.annotation_metadata = jams.AnnotationMetadata(
        corpus=corpus,
        version=version,
        annotation_tools=annotation_tools,
        annotator=annotator,
        data_source=data_source,
        curator=jams.Curator(name='Michaël Defferrard', email='michael.defferrard@epfl.ch'),
    )
    return tag


if __name__ == '__main__':
    parse()
