import os
import sys
import warnings
from os import walk
from os.path import splitext, join
from statistics import median, stdev, mean
from typing import List, Tuple

import audioread
import jams
import numpy as np
import pkg_resources
import pybtex.database


def get_bibtex_entry(cite_key):
    references_file = pkg_resources.resource_filename('tempo_eval', 'references.bib')
    database = pybtex.database.parse_file(references_file)
    if cite_key in database.entries:
        entry = database.entries[cite_key]
        entry_database = pybtex.database.BibliographyData()
        entry_database.add_entry(cite_key, entry)
        return entry_database.to_string('bibtex')
    else:
        return None


def create_jam(annotation_file_name, audio_base_dir, artist=None, title=None, release=None, duration=None):
    jam = jams.JAMS()
    if artist is not None:
        jam.file_metadata.artist = artist
    if title is not None:
        jam.file_metadata.title = title
    if release is not None:
        jam.file_metadata.release = release
    if duration is not None:
        jam.file_metadata.duration = duration
    elif audio_base_dir is not None and duration is None:
        base = get_base(annotation_file_name)
        for (dirpath, _, filenames) in walk(audio_base_dir):
            audio_file_name, duration = _get_duration(filenames, base, dirpath)
            if duration is not None:
                jam.file_metadata.identifiers = {'file': audio_file_name}
                jam.file_metadata.duration = duration
                break
    if jam.file_metadata.duration is None:
        warnings.warn('Failed to find duration for file {}'.format(annotation_file_name))
    return jam


def get_base(file):
    base, _ = splitext(file)
    # hack to also get rid of a 2nd extension, as they are used in RWC
    if file.endswith('.BEAT.TXT'):
        base, _ = splitext(base)
    elif file.endswith('.wav.txt'):
        base, _ = splitext(base)
    # hack to also get rid of a 2nd extension, as they are produced by madmom
    elif file.endswith('.bpm.txt'):
        base, _ = splitext(base)
    # hack to also get rid of ismir2004 song extensions
    while '.txt' in base:
        base, _ = splitext(base)
    return base


def create_tempo_annotation(tempo1, tempo2=0.0, confidence1=1.0, license=None):
    tempo = jams.Annotation(namespace='tempo')
    if license:
        tempo.sandbox = {'license': license}
    tempo.append(time=0.0,
                 duration='nan',
                 value=tempo1,
                 confidence=confidence1)
    if tempo2 != 0.0:
        tempo.append(time=0.0,
                     duration='nan',
                     value=tempo2,
                     confidence=1.0 - confidence1)
    return tempo


def create_beat_annotation(beats, license=None):
    beat = jams.Annotation(namespace='beat')
    if license:
        beat.sandbox = {'license': license}
    for b in beats:
        beat.append(time=b['timestamp'], value=b['position'], duration='nan', confidence=b['confidence'])
    return beat


def create_tag_open_annotation(*tag_names, confidence=1.0, license=None):
    tag = jams.Annotation(namespace='tag_open')
    if license:
        tag.sandbox = {'license': license}
    for tag_name in set(tag_names):
        if len(tag_name.strip()) > 0:
            tag.append(time=0.0, value=tag_name, duration='nan', confidence=confidence)
    return tag


def create_tag_fma_genre_annotation(*tag_names, confidence=1.0, license=None):
    tag = jams.Annotation(namespace='tag_fma_genre')
    if license:
        tag.sandbox = {'license': license}
    for tag_name in set(tag_names):
        if len(tag_name.strip()) > 0:
            tag.append(time=0.0, value=tag_name, duration='nan', confidence=confidence)
    return tag


def _get_duration(file_names, base, dirpath):
    for audio_file_name in [file_name for file_name in file_names if (os.sep + base) in join(dirpath, file_name)]:
        try:
            with audioread.audio_open(join(dirpath, audio_file_name)) as file:
                duration = file.duration
                return audio_file_name, duration
        except:
            print('Failed to read audio file {}'.format(audio_file_name), file=sys.stderr)
    return None, None


def timestamps_to_bpm(timestamps: List[float], meter: int = 1) -> Tuple[float, float, float, float]:
    if meter < 1:
        raise ValueError('meter must be greater than 0: '.format(meter))
    ibis = []
    for position in range(meter):
        corresponding = corresponding_timestamps(timestamps, meter, position)
        ibis.extend([inter_beat_interval_to_bpm(timestamp, meter=meter) for timestamp in inter_timestamp_intervals(corresponding)])
        # ibis.extend([timestamp for timestamp in inter_beat_intervals(corresponding)])
    if len(ibis) == 0:
        print('No IBIs found for timestamps {}'.format(timestamps), file=sys.stderr)
    ibis_median = median(ibis)
    ibis_mean = mean(ibis)
    ibis_stdev = stdev(ibis)
    #ibis_c_var = ibis_stdev/ibis_mean
    c_var = float(np.std(np.array(ibis) / ibis_mean))
    return ibis_median, ibis_mean, ibis_stdev, c_var


def inter_beat_interval_to_bpm(seconds, meter=1):
    return (60.0 / seconds) * meter


def inter_timestamp_intervals(timestamps):
    ibis = []
    for i in range(len(timestamps)-1):
        ibis.append(timestamps[i+1] - timestamps[i])
    return ibis


def corresponding_timestamps(timestamps, meter, offset):
    corresponding = []
    for i in range(offset, len(timestamps), meter):
        corresponding.append(timestamps[i])
    return corresponding


def create_annotator(bibtex):
    annotator = {}
    if bibtex:
        entries = []
        if isinstance(bibtex, str):
            entry = get_bibtex_entry(bibtex)
            if entry:
                entries.append(entry)
            else:
                print('Failed to find bibtex entry for cite-key \'{}\'. '
                      'Please make sure it occurs in file tempo_eval/references.bib.'.format(bibtex))
        else:
            for b in bibtex:
                entry = get_bibtex_entry(b)
                if entry:
                    entries.append(entry)
                else:
                    print('Failed to find bibtex entry for cite-key \'{}\'. '
                          'Please make sure it occurs in file tempo_eval/references.bib.'.format(bibtex))
        if len(entries) == 1:
            annotator = {'bibtex': entries[0]}
        elif len(entries) > 1:
            annotator = {'bibtex': entries}
    return annotator