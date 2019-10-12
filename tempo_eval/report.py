# -*- coding: utf-8 -*-
"""
Report generator.

This module primarily deals with formatting and printing/writing.
"""
import csv
import getpass
import logging
import operator
import random
import time
from collections import OrderedDict
from enum import IntEnum
from math import isfinite
from os import remove
from os.path import join, basename, dirname, relpath
from statistics import mean, stdev
from typing import Any, Tuple, Dict, List, Iterable, Optional, Set, Callable, Union

import jams
import markdown as mdown
import numpy as np
import pandas as pd
import pkg_resources
import pybtex.database
from jams.util import smkdirs
from pygam import LinearGAM

from tempo_eval.evaluation import read_reference_annotations, significant_difference, \
    item_ids_for_differing_annotations, fraction_lt_c_var, read_estimate_annotations, \
    items_lt_c_var, basic_statistics, list_reference_corpus_names, \
    extract_c_var_from_beats, extract_tempo, items_in_tempo_intervals, \
    extract_tags, items_per_tag, list_estimate_corpus_names, \
    ACC1, ACC2, APE1, APE2, PE2, PE1, OE2, OE1, PSCORE, ONE_CORRECT, BOTH_CORRECT, \
    read_annotations, Annotations, TagAnnotations, Metric, \
    EvalResults, AverageResults, AOE1, AOE2, mcnemar
from tempo_eval.figures import render_horizontal_bar_chart, render_line_graph, render_violin_plot, \
    render_tag_violin_plot
from tempo_eval.markdown_writer import MarkdownWriter
from tempo_eval.version import version

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="description" content="Tempo estimation evaluation report by tempo_eval.">
    <meta name="keywords" content="MIR music information retrieval tempo estimation evaluation">
    <meta name="generator" content="tempo_eval $VERSION">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
        integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T"
        crossorigin="anonymous">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <style>
        body {
            font-family: sans-serif;
        }
        code, pre {
            font-family: monospace;
        }
        h1, h2, h3 {
            margin-top: 1.5em;
        }
        h4, h5, h6 {
            margin-top: 1em;
        }
        h1 code,
        h2 code,
        h3 code,
        h4 code,
        h5 code,
        h6 code {
            font-size: inherit;
        }
        div.container {
            width: 75%;
            margin: 0 auto;
        }
        figure {
            text-align: center;
        }
        embed {
            max-width: 800px;
        }
        tr:nth-child(even) {
            background: #FFF;
        }
        thead {
            border-top: 2px solid #BBB;
            border-bottom: 1px solid #BBB;
        }
        tbody {
            border-bottom: 2px solid #BBB;
        }
        td, th {
            padding:0 15px 0 15px;
        }
        </style>
</head>
<body>
<div class="container">
$CONTENT
</div>
</body>
</html>
"""


class Size(IntEnum):
    """
    Report sizes.
    """
    S = 0      #: Report size "small".
    M = 50     #: Report size "medium".
    L = 100    #: Report size "large".
    XL = 150   #: Report size "extra large". Reserved for experimental metrics.


def print_report(output_dir: str = './',
                 validate: bool = True,
                 corpus_names: List[str] = None,
                 estimates_dir: str = None,
                 references_dir: str = None,
                 include_relative_dir_in_item_id: bool = True,
                 format: str = 'kramdown',
                 size: Size = Size.L) -> str:
    """
    Print reports for references and estimates and make
    them accessible via an index page.

    By default the report covers all built-in references and estimates.
    When providing an estimates or reference directory, which must
    contain jams, the report covers only corpora that occur in the
    provided annotations or are specified in the ``corpus_names`` argument.
    By convention, tempo_eval uses the file's :py:func:`os.path.relpath`
    as item id (relative to the estimates or references directory).
    You can change this behavior to use :py:func:`os.path.basename`
    by setting ``include_relative_dir_in_item_id`` to ``False``.
    You can override either of these two mechanisms by adding a
    ``jam.file_metadata.identifiers['tempo_eval_id']``-value to your jams.

    :param size: size of evaluation report---choose "L" for a
        comprehensive version, choose "S" for a shorter version
        with just the essentials
    :param format: ``kramdown``, ``markdown`` or ``html``
    :param corpus_names: list of corpora to generate reports for.
        See :py:func:`~tempo_eval.list_reference_corpus_names` to get a list of valid names.
    :param output_dir: output directory
    :param validate: validate jam while reading (has major performance impact)
    :param references_dir: directory to search for reference jams
    :param estimates_dir: directory to search for estimate jams
    :param include_relative_dir_in_item_id: when using an extra reference or
        estimate directory, use the relative directory name as part of the
        item id or not.
    :return: output file name for the index page
    :raises ValueError: if the given format is not supported

    .. note:: By default, `validate` is `True` in order to stay safe.
        But since it affects performance quite a bit, you might want to turn
        validation off when using this function to keep your sanity.

    """

    if format not in ['kramdown', 'markdown', 'html']:
        raise ValueError('Unsupported output format: {}'.format(format))

    corpora = []
    if corpus_names:
        corpora.extend(corpus_names)
        logging.debug('Explicitly specified corpora: {}'.format(corpus_names))

    additional_estimates = {}
    if estimates_dir is not None:
        item_id_fn = _derive_item_id_fn(estimates_dir, include_relative_dir_in_item_id)
        additional_estimates = read_annotations(estimates_dir,
                                                namespace='tempo',
                                                validate=validate,
                                                derive_item_id=item_id_fn,
                                                split_by_corpus=True)
        found_corpora = list(additional_estimates.keys())
        corpora.extend(found_corpora)
        logging.debug('Found corpora in specified estimates: {}'.format(found_corpora))

    additional_references = {}
    if references_dir is not None:
        item_id_fn = _derive_item_id_fn(references_dir, include_relative_dir_in_item_id)
        additional_references = read_annotations(references_dir,
                                                 namespace='tempo',
                                                 validate=validate,
                                                 derive_item_id=item_id_fn,
                                                 split_by_corpus=True)
        found_corpora = list(additional_references.keys())
        corpora.extend(found_corpora)
        logging.debug('Found corpora in specified references: {}'.format(found_corpora))

    # fall back to built-in corpus references
    if not corpora:
        corpora = list_reference_corpus_names()
        corpora.extend(list_estimate_corpus_names())
        logging.debug('Found corpora in built-in annotations: {}'.format(corpora))

    corpora = sorted(list(set(corpora)))
    if not corpora:
        logging.error('No corpora found.')
        return None

    logging.debug('Effective corpora: {}'.format(corpora))

    corpus_files = []
    for corpus_name in corpora:
        # check for additional annotations for this corpus
        add_ref = additional_references[corpus_name]\
            if corpus_name in additional_references\
            else None
        add_est = additional_estimates[corpus_name]\
            if corpus_name in additional_estimates\
            else None
        corpus_report = print_corpus_report(corpus_name,
                                            output_dir=output_dir,
                                            validate=validate,
                                            additional_references=add_ref,
                                            additional_estimates=add_est,
                                            format=format,
                                            size=size)
        corpus_files.append(corpus_report)

    smkdirs(output_dir)
    file_name = join(output_dir, 'index.md')
    with open(file_name, "w", encoding='UTF-8') as f:
        jekyll = format == 'kramdown'
        kramdown = format == 'kramdown'
        md = MarkdownWriter(f, jekyll=jekyll, kramdown=kramdown)
        md.h1('tempo_eval')
        md.paragraph('Detailed evaluations for the following corpora:')
        for corpus_name, corpus_file in zip(corpora, corpus_files):
            md.write('- ')
            md.link(corpus_name, basename(corpus_file))
            md.writeln()
        md.writeln()

        if size >= Size.L:
            _print_all_corpora_tempo_variation(md, corpora, validate, output_dir=output_dir)

        _print_generation_date(md, size=size)

    if format == 'html':
        file_name = convert_md_to_html(file_name)

    logging.info('Done. The main report page can be found at {}'.format(file_name))
    return file_name


def print_corpus_report(corpus_name: str,
                        output_dir: str = './',
                        validate: bool = True,
                        additional_estimates: Dict[str, Annotations] = None,
                        additional_references: Dict[str, Annotations] = None,
                        format: str = 'kramdown',
                        size: Size = Size.L) -> str:
    """
    Print report for the given named corpus.

    :param size: Size of evaluation report. Choose "L" for the
        most comprehensive version, choose "S" for a shorter version
        with just the essentials.
    :param format: ``kramdown``, ``markdown`` or ``html``
    :param corpus_name: corpus name.
        See :py:func:`~tempo_eval.list_reference_corpus_names` to get a list of valid names.
    :param output_dir: output directory
    :param validate: validate jam while reading (has major performance impact)
    :param additional_references: additional, user-provided reference annotations
    :param additional_estimates: additional, user-provided estimate annotations
    :return: output file name
    :raises ValueError: if the given format is not supported

    .. note:: By default, `validate` is `True` in order to stay safe.
        But since it affects performance quite a bit, you might want to turn
        validation off when using this function to keep your sanity.

    """

    if format not in ['kramdown', 'markdown', 'html']:
        raise ValueError('Unsupported output format: {}'.format(format))

    logging.info('Creating report for \'{}\'...'.format(corpus_name))
    logging.debug('Loading annotations...')

    def merge_annotations(existing_annotations, additional_annotations):

        merged = {}
        namespaces = set(list(existing_annotations.keys())
                         + list(additional_annotations.keys()))
        for ns in namespaces:
            if ns not in existing_annotations or not existing_annotations[ns]:
                merged[ns] = additional_annotations[ns]
                continue
            if ns not in additional_annotations or not additional_annotations[ns]:
                merged[ns] = existing_annotations[ns]
                continue

            # namespace occurs in both, let's see...
            ex = existing_annotations[ns]
            ad = additional_annotations[ns]
            merged[ns] = ex
            for version, items in ad.items():
                if version in ex:
                    new_version = '{}_{}'.format(getpass.getuser(), version)\
                        .replace(' ', '_')
                    if new_version in ex:
                        new_version = '{}_{}_{}'\
                            .format(getpass.getuser(),
                                    version, random.randint(10000, 99999))\
                            .replace(' ', '_')
                    logging.info('Annotations with version \'{}\' already '
                                 'exists and will be replaced with \'{}\'.'
                                 .format(version, new_version))
                    ex[new_version] = ad[version]
                else:
                    ex[version] = ad[version]
        return merged

    estimates = {}
    try:
        estimates = read_estimate_annotations(corpus_name, validate=validate)
        logging.info('Found built-in estimates: {}'.format(list(estimates.keys())))
    except FileNotFoundError as fnfe:
        logging.error('Failed to read built-in estimates for corpus \'{}\' ({})'
                      .format(corpus_name, fnfe))

    if additional_estimates:
        estimates = merge_annotations(estimates, additional_estimates)

    tempo_estimates = estimates['tempo'] if 'tempo' in estimates else {}

    # read multiple kinds of annotations ("namespaces") in one go
    # in order to avoid reading and parsing jams multiple times.
    references = {}
    try:
        references = read_reference_annotations(corpus_name,
                                                namespace=['tag_open', 'tag_gtzan',
                                                           'tag_fma_genre', 'tempo',
                                                           'beat'],
                                                validate=validate)
        logging.info('Found built-in references: {}'.format(list(references.keys())))
    except FileNotFoundError as fnfe:
        logging.error('Failed to read built-in references for corpus \'{}\' ({})'
                      .format(corpus_name, fnfe))

    if additional_references:
        references = merge_annotations(references, additional_references)
    tempo_references = references['tempo'] if 'tempo' in references else {}
    beat_references = references['beat'] if 'beat' in references else {}
    tag_references = {}
    if 'tag_open' in references:
        tag_references['tag_open'] = references['tag_open']
    if 'tag_gtzan' in references:
        tag_references['tag_gtzan'] = references['tag_gtzan']
    if 'tag_fma_genre' in references:
        tag_references['tag_fma_genre'] = references['tag_fma_genre']

    logging.debug('Annotations loaded.')

    smkdirs(output_dir)
    file_name = join(output_dir, '{}.md'.format(corpus_name
                                                .replace(' ', '_')
                                                .replace('/', '_')
                                                .replace('\\', '_')
                                                .replace(':', '')
                                                .replace(';', '')))
    with open(file_name, "w", encoding='utf-8') as f:

        jekyll = format == 'kramdown'
        kramdown = format == 'kramdown'
        md = MarkdownWriter(f, kramdown=kramdown, jekyll=jekyll)

        html = format == 'html'
        if not html:
            md.h1(corpus_name + '\n{:.no_toc}')
        else:
            md.h1(corpus_name)

        md.paragraph('This is the tempo_eval report for the \'{}\' corpus.'.format(corpus_name))

        if not html:
            # the following line will be replaced by kramdown
            # see https://kramdown.gettalong.org/converter/html.html#toc
            md.h2('Table of Contents' + '\n{:.no_toc}')
            md.paragraph('- TOC\n{:toc}')
        else:
            md.h2('Table of Contents')
            md.paragraph('[TOC]')  # will be replace by toc markdown extension

        if not tempo_references and len(tempo_estimates) > 1:
            if len(tempo_estimates) == 2:
                # we have two estimates, but no reference.
                # pick one as reference (sort alphanumerically, pick first)
                tempo_references, tempo_estimates = _alphanumeric_split(tempo_estimates)
                md.blockquote('Because reference annotations are '
                              'not available, we treat the *estimate* {} '
                              'as reference.'
                              .format(md.to_headline_link(list(tempo_references.keys())[0])))
            else:
                # we have 2 or more estimates, pick the one
                # with the highest average agreement as reference
                # create accuracy1 matrix (not efficient)
                tempo_references, tempo_estimates = _highest_mma_split(tempo_estimates)
                md.blockquote('Because reference annotations are not '
                              'available, we treat the *estimate* {} as '
                              'reference. It has the highest Mean '
                              'Mutual Agreement (MMA), based on Accuracy1 with '
                              '4% tolerance.'
                              .format(md.to_headline_link(list(tempo_references.keys())[0])))

        if tempo_references:
            _print_reference_eval(md, corpus_name, tempo_references,
                                  beat_references, tag_references,
                                  output_dir=output_dir, size=size)
        else:
            md.blockquote('No reference Jams found.')
            logging.warning('No reference Jams found for corpus {}'.format(corpus_name))
        if tempo_estimates:
            _print_estimates_eval(md, corpus_name, tempo_references, tempo_estimates,
                                  beat_references, tag_references, output_dir=output_dir,
                                  size=size)
        else:
            md.blockquote('No estimate Jams found.')
            logging.warning('No estimate Jams found for corpus {}'.format(corpus_name))
        _print_generation_date(md, size=size)

    if format == 'html':
        file_name = convert_md_to_html(file_name)

    return file_name


def _print_reference_eval(md: MarkdownWriter,
                          corpus_name: str,
                          tempo_references: Annotations,
                          beat_references: Annotations,
                          tag_references: TagAnnotations,
                          output_dir: str = './',
                          size: Size = Size.L) -> None:
    """
    Basic report part for reference annotations.

    :param md: markdown
    :param corpus_name: corpus name
    :param tempo_references: tempo reference annotations
    :param beat_references: beat reference annotations
    :param tag_references: tag reference annotations
    :param output_dir: output directory
    :param size: size
    """

    logging.info('Printing reference report...')

    md.h1('References for \'{}\''.format(corpus_name))
    md.h2('References')
    _print_annotation_metadata(md, tempo_references, output_dir=output_dir)

    base_name = '{}_reference'.format(corpus_name)
    _print_basic_statistics(md, '{}_basic_stats'.format(base_name), tempo_references,
                            estimates=False, output_dir=output_dir)
    if size >= Size.M:
        _print_percentage_over_interval(md, base_name, tempo_references, output_dir=output_dir)
        _print_percentage_for_tag(md, base_name, tag_references, output_dir=output_dir)
    if beat_references and size >= Size.L:
        _print_tempo_variation(md, corpus_name, beat_references, output_dir=output_dir)


def _print_tempo_variation(md: MarkdownWriter,
                           corpus_name: str,
                           beat_references: Annotations,
                           output_dir: str = './') -> None:
    """
    Print line graph for fraction of dataset with certain tempo variation
    (measured in CV).

    :param md: markdown
    :param corpus_name: corpus name
    :param beat_references: beat reference annotations
    :param output_dir: output directory
    """
    md.h2('Beat-Based Tempo Variation')
    # get beat annotations and show percentage of dataset at different cv
    # list 0.05 and 0.1
    cv_tresholds = [round(t, 4) for t in np.arange(0, 0.505, 0.005)]
    cvs = extract_c_var_from_beats(beat_references)
    fractions_lt_cv = fraction_lt_c_var(cvs, thresholds=cv_tresholds)
    legend = list(fractions_lt_cv.keys())
    y_values = {l: fractions_lt_cv[l] for l in legend}

    values_df = pd.DataFrame(y_values, index=cv_tresholds)
    values_df.name = 'Fraction of Dataset Below Coefficient of Variation-Threshold'
    values_df.index.name = 'τ'

    _print_line_graph(md, '{}_variation'.format(corpus_name), values_df,
                      caption='Fraction of the dataset with beat-annotated tracks with c<sub>var</sub> < τ.',
                      y_axis_label='Fraction of Dataset (%)', output_dir=output_dir)


def _print_all_corpora_tempo_variation(md: MarkdownWriter,
                                     reference_names: Iterable[str],
                                     validate: bool,
                                     output_dir: str = './') -> None:
    """
    Print diagram displaying tempo variations in all reference corpora.

    :param md: markdown writer
    :param reference_names: reference dataset names
    :param validate: validate jams?
    :param output_dir: output directory
    """
    md.h1('Beat-Based Tempo Variation')
    y_values = {}
    cv_tresholds = [round(t, 4) for t in np.arange(0, 0.505, 0.005)]
    for corpus_name in reference_names:

        try:
            annotations = read_reference_annotations(corpus_name,
                                                     namespace='beat',
                                                     validate=validate)
            if 'beat' in annotations:
                beat_references = annotations['beat']
                # get beat annotations and show percentage of dataset at different cv
                # list 0.05 and 0.1
                fractions_lt_c_var = fraction_lt_c_var(extract_c_var_from_beats(beat_references),
                                                                thresholds=cv_tresholds)
                y_values.update({'{} {}'.format(corpus_name, l): fractions_lt_c_var[l]
                                 for l in fractions_lt_c_var.keys()})
        except FileNotFoundError as fnfe:
            logging.error('Failed to read built-in beat estimates for corpus \'{}\' ({})'
                          .format(corpus_name, fnfe))

    values_df = pd.DataFrame(y_values, index=cv_tresholds)
    values_df.name = 'Fraction of Dataset Below Coefficient of Variation-Threshold'
    values_df.index.name = 'τ'

    _print_line_graph(md, 'all_variation', values_df,
                      caption='Fraction of the dataset with beat-annotated '
                              'tracks, where beats/track fall below a given '
                              'coefficient of variation τ.',
                      y_axis_label='Fraction of Dataset (%)',
                      output_dir=output_dir)


def _print_estimates_eval(md: MarkdownWriter,
                          corpus_name: str,
                          tempo_references: Annotations,
                          tempo_estimates: Annotations,
                          beat_references: Annotations,
                          tag_references: TagAnnotations,
                          output_dir: str = './',
                          size: Size = Size.L) -> None:
    """
    Print basic statistics and descriptions for the estimates and
    an evaluation for the estimates w.r.t. reference values.

    :param md: markdown writer
    :param corpus_name: corpus name
    :param tempo_references: tempo references
    :param tempo_estimates: tempo estimates
    :param beat_references: beat references
    :param tag_references: tag references
    :param output_dir: output_dir
    """

    base_name = '{}_estimates'.format(corpus_name)

    logging.info('Printing estimates report...')

    md.h1('Estimates for \'{}\''.format(corpus_name))
    md.h2('Estimators')
    _print_annotation_metadata(md, tempo_estimates, output_dir=output_dir)

    _print_basic_statistics(md, '{}_basic_stats'.format(base_name),
                            tempo_estimates,
                            estimates=True,
                            output_dir=output_dir)

    if size >= Size.M:

        _print_percentage_over_interval(md, base_name,
                                        tempo_estimates,
                                        output_dir=output_dir)

    logging.debug('Printing eval...')

    # only eval, when there's something to compare, i.e. we have both estimates and references
    if tempo_estimates and tempo_references:

        _print_accuracy_evaluation(md, base_name, tempo_references,
                                   tempo_estimates, beat_references,
                                   tag_references,
                                   output_dir=output_dir,
                                   size=size)

        _print_mirex_evaluation(md, base_name, tempo_references,
                                tempo_estimates, beat_references,
                                tag_references,
                                output_dir=output_dir,
                                size=size)

        _print_error_metric_evaluation(md,
                                       OE1,
                                       OE2,
                                       base_name, tempo_references,
                                       tempo_estimates, beat_references,
                                       tag_references,
                                       output_dir=output_dir,
                                       size=size)

        _print_error_metric_evaluation(md,
                                       AOE1,
                                       AOE2,
                                       base_name, tempo_references,
                                       tempo_estimates, beat_references,
                                       tag_references,
                                       output_dir=output_dir,
                                       size=size)

        if size >= Size.XL:
            _print_error_metric_evaluation(md,
                                           PE1,
                                           PE2,
                                           base_name, tempo_references,
                                           tempo_estimates, beat_references,
                                           tag_references,
                                           output_dir=output_dir,
                                           size=size)

            _print_error_metric_evaluation(md,
                                           APE1,
                                           APE2,
                                           base_name, tempo_references,
                                           tempo_estimates, beat_references,
                                           tag_references,
                                           output_dir=output_dir,
                                           size=size)


def _print_error_metric_evaluation(md: MarkdownWriter,
                                   metric1: Metric,
                                   metric2: Metric,
                                   base_name: str,
                                   tempo_references: Annotations,
                                   tempo_estimates: Annotations,
                                   beat_references: Annotations,
                                   tag_references: TagAnnotations,
                                   output_dir: str = './',
                                   size: Size = Size.L) -> None:
    """
    Print an evaluation based on an error metric like PE.

    :param md: markdown writer
    :param metric1: ape1 or pe1 or...
    :param metric2: ape2 or pe2 or...
    :param base_name: base name for files (e.g. CSV export)
    :param tempo_references: tempo references
    :param tempo_estimates:  tempo estimates
    :param beat_references: beat references
    :param tag_references: tag references
    :param output_dir: output dir
    :param size: size
    """

    logging.debug('Printing {}/{}...'.format(metric1.name, metric2.name))

    md.h2('{} and {}'.format(metric1.formatted_name, metric2.formatted_name))
    md.paragraph(metric1.description)
    md.paragraph(metric2.description)

    reference_tempi = metric1.extract_tempi(tempo_references)
    estimated_tempi = metric2.extract_tempi(tempo_estimates)

    results1 = metric1.eval_tempi(reference_tempi, estimated_tempi)
    results2 = metric2.eval_tempi(reference_tempi, estimated_tempi)

    averages1 = metric1.averages(results1)
    averages2 = metric2.averages(results2)

    for ref_name in sorted(averages1.keys()):

        _print_mean_stdev_table(md, base_name, metric1, metric2,
                                results1, results2,
                                averages1, averages2,
                                ref_name, output_dir=output_dir)

        _print_error_distribution(md, base_name,
                                  ref_name,
                                  results1,
                                  metric1,
                                  output_dir=output_dir)

        _print_error_distribution(md, base_name,
                                  ref_name,
                                  results2,
                                  metric2,
                                  output_dir=output_dir)

        if size >= Size.XL:
            if not metric1.signed:
                _print_mean_metric_results(md, base_name,
                                           ref_name,
                                           averages1,
                                           metric1,
                                           output_dir=output_dir)

            if not metric2.signed:
                _print_mean_metric_results(md, base_name,
                                           ref_name,
                                           averages2,
                                           metric2,
                                           output_dir=output_dir)

    if len(tempo_estimates) > 1 and size >= Size.M \
            and (metric1.significant_difference_function is not None
                 or metric2.significant_difference_function is not None):
        # pick out significance for tolerance = 0.04
        md.h3('Significance of Differences')
        # square table with pvalues
        if metric1.significant_difference_function is not None:
            _print_significance_matrix(md, base_name, results1, metric1,
                                       output_dir=output_dir)
        if metric2.significant_difference_function is not None:
            _print_significance_matrix(md, base_name, results2, metric2,
                                       output_dir=output_dir)

    if size >= Size.L:
        _print_subset_evaluations(md, base_name,
                                  [metric1, metric2],
                                  [results1, results2],
                                  tempo_references,
                                  tempo_estimates,
                                  beat_references,
                                  tag_references,
                                  output_dir=output_dir)


def _print_error_distribution(md: MarkdownWriter, base_name: str,
                              ref_name: str,
                              results: EvalResults,
                              metric: Metric,
                              output_dir: str = './'):
    """
    Print error distribution as descriptive violin plot.

    :param md: markdown writer
    :param base_name: base name for created files
    :param ref_name: name of reference annotations
    :param results: results
    :param metric: metric
    :param output_dir: output directory
    """

    md.h3('{} distribution for {}'.format(metric.formatted_name, ref_name))

    res = {name: [i[0] for i in results[ref_name][name].values()] for name in results[ref_name].keys()}
    res = OrderedDict(sorted(res.items(), key=lambda x: x[0]))

    scores_df = pd.DataFrame.from_dict(res)
    scores_df.name = '{} for {}'.format(metric.name, ref_name)
    scores_df.index.name = 'Estimator'

    axis_label = '{}'.format(metric.name)
    if metric.unit:
        axis_label = '{} ({})'.format(metric.name, metric.unit)

    _print_violin_plot(md, '{}_{}_distribution_{}'.format(base_name, ref_name, metric.name.lower()),
                       scores_df,
                       caption='{metric} for estimates compared to version {ref}. Shown are the mean '
                               '{metric} and an empirical distribution of the sample, '
                               'using kernel density estimation (KDE).'
                       .format(metric=metric.formatted_name, ref=md.to_headline_link(ref_name)),
                       axis_label=axis_label,
                       output_dir=output_dir)


def _print_mean_metric_results(md: MarkdownWriter, base_name: str,
                               ref_name: str,
                               average_results: AverageResults,
                               metric: Metric,
                               output_dir: str = './'):
    """
    Print mean metric results.

    :param md: markdown writer
    :param base_name: base name for created files
    :param ref_name: name of reference annotations
    :param results: mean results
    :param metric: metric
    :param output_dir: output directory
    """

    md.h3('Mean {} for {}'.format(metric.formatted_name, ref_name))

    results = {name: [average_results[ref_name][name][0][0]]
               for name in average_results[ref_name].keys()}
    results = OrderedDict(sorted(results.items(), key=lambda x: x[0]))

    stdev = {name: [average_results[ref_name][name][1][0]]
             for name in average_results[ref_name].keys()}
    stdev = OrderedDict(sorted(stdev.items(), key=lambda x: x[0]))

    scores_df = pd.DataFrame.from_dict(results, orient='index', columns=[metric.name])
    scores_df.name = 'Mean {} for {}'.format(metric.name, ref_name)
    scores_df.index.name = 'Estimator'

    stdev_df = pd.DataFrame.from_dict(stdev, orient='index', columns=['stdev'])
    stdev_df.index.name = 'Estimator'

    x_axis_label = 'Mean {}'.format(metric.name)
    if metric.unit:
        x_axis_label = 'Mean {} ({})'.format(metric.name, metric.unit)

    _print_horizontal_bar_chart(md, '{}_{}_mean_{}'.format(base_name, ref_name, metric.name.lower()),
                                scores_df,
                                caption='Mean {} compared to version {}.'
                                .format(metric.formatted_name, md.to_headline_link(ref_name)),
                                x_axis_label=x_axis_label,
                                output_dir=output_dir)


def _print_mirex_evaluation(md: MarkdownWriter,
                            base_name: str,
                            tempo_references: Annotations,
                            tempo_estimates: Annotations,
                            beat_references: Annotations,
                            tag_references: TagAnnotations,
                            output_dir: str = './',
                            size: Size = Size.L) -> None:
    """
    Print `MIREX <https://www.music-ir.org/mirex/wiki/>`_-style evaluation,
    i.e., p-score, one correct, both correct and 8% tolerance.

    .. seealso:: `MIREX Audio Tempo Extraction 2006
        <https://www.music-ir.org/mirex/wiki/2006:Audio_Tempo_Extraction>`_

    .. seealso:: McKinney, M. F., Moelants, D., Davies, M. E., and Klapuri, A. P. (2007).
        `Evaluation of audio beat tracking and music tempo extraction algorithms.
        <http://www.cs.tut.fi/sgn/arg/klap/mckinney_jnmr07.pdf>`_
        Journal of New Music Research, 36(1):1–16.

    :param md: markdown writer
    :param base_name: base name for files to be created
    :param tempo_references: tempo references
    :param tempo_estimates: tempo estimates
    :param beat_references: beat references
    :param tag_references: tag references
    :param output_dir: output dir
    :param size: size
    """

    logging.debug('Printing MIREX...')

    def _any_reference_tempi_suitable(reference_tempi):
        mirex_style = False
        for ref in reference_tempi.values():
            if PSCORE.are_tempi_suitable(ref):
                mirex_style = True
                break
        return mirex_style

    reference_tempi = PSCORE.extract_tempi(tempo_references)

    if _any_reference_tempi_suitable(reference_tempi):

        md.h2('MIREX-Style Evaluation')
        md.paragraph(PSCORE.description)
        md.paragraph(ONE_CORRECT.description)
        md.paragraph(BOTH_CORRECT.description)
        bib_file_name, cite_key = _extract_bibtex_entry(output_dir, 'McKinney2007')
        md.paragraph('See [[{cite_key}]({bibtex})].'.format(cite_key=cite_key,
                                                            bibtex=dir_filename(bib_file_name)))

        md.paragraph('Note: Very few datasets actually have multiple '
                     'annotations per track along with a salience distributions. '
                     'References without suitable annotations are not shown.',
                     emphasize=True)

        estimated_tempi = PSCORE.extract_tempi(tempo_estimates)

        standard_tolerance = 0.08
        if size >= Size.M:
            tolerances = [round(t, 5) for t in np.arange(0.0, 0.101, 0.001)]
        else:
            tolerances = [standard_tolerance]

        mirex_metrics = [PSCORE, ONE_CORRECT, BOTH_CORRECT]
        mirex_averages = []
        mirex_results = []

        for metric in mirex_metrics:
            eval_results = metric.eval_tempi(reference_tempi,
                                             estimated_tempi,
                                             tolerances=tolerances)
            mirex_results.append(eval_results)
            mirex_averages.append(metric.averages(eval_results))

        for ref_name in sorted(mirex_averages[0].keys()):
            if not PSCORE.are_tempi_suitable(reference_tempi[ref_name]):
                continue

            _print_mirex_table(md, base_name,
                               mirex_metrics,
                               mirex_results,
                               mirex_averages,
                               ref_name,
                               tolerances,
                               standard_tolerance,
                               output_dir=output_dir)

            if len(tolerances) > 1:
                for metric, result in zip(mirex_metrics, mirex_averages):

                    _print_tolerance_chart(md, base_name,
                                           metric,
                                           ref_name,
                                           result,
                                           tolerances,
                                           output_dir=output_dir)

        # should we really offer McNemar/t-test for PSCORE, ONE_CORRECT, and BOTH_CORRECT??
        # if len(tempo_estimates) > 1 and size >= Size.M \
        #         and (PSCORE.significant_difference_function is not None
        #              or ONE_CORRECT.significant_difference_function is not None
        #              or BOTH_CORRECT.significant_difference_function is not None):
        #     # pick out significance for tolerance = 0.04
        #     md.h3('Significance of Differences')
        #     # square table with pvalues
        #
        #     for i, metric in enumerate(mirex_metrics):
        #         if metric.significant_difference_function is not None:
        #             _print_significance_matrix(md, base_name, mirex_results[i], metric,
        #                                        output_dir=output_dir)

        if size >= Size.L:
            _print_subset_evaluations(md, base_name,
                                      [PSCORE, ONE_CORRECT, BOTH_CORRECT],
                                      None,
                                      tempo_references,
                                      tempo_estimates,
                                      beat_references,
                                      tag_references,
                                      output_dir=output_dir)


def _print_accuracy_evaluation(md: MarkdownWriter,
                               base_name: str,
                               tempo_references: Annotations,
                               tempo_estimates: Annotations,
                               beat_references: Annotations,
                               tag_references: TagAnnotations,
                               output_dir: str = './',
                               size: Size = Size.L) -> None:
    """
    Print evaluation using the traditional accuracy
    metrics accuracy 1 and accuracy 2.

    .. seealso:: Fabien Gouyon, Anssi P. Klapuri, Simon Dixon, Miguel Alonso,
        George Tzanetakis, Christian Uhle, and Pedro Cano. `An experimental
        comparison of audio tempo induction algorithms.
        <https://www.researchgate.net/profile/Fabien_Gouyon/publication/3457642_An_experimental_comparison_of_audio_tempo_induction_algorithms/links/0fcfd50d982025360f000000/An-experimental-comparison-of-audio-tempo-induction-algorithms.pdf>`_
        IEEE Transactions on Audio, Speech, and Language Processing,
        14(5):1832– 1844, 2006.

    :param md: markdown writer
    :param base_name: base name for created files
    :param tempo_references: tempo references
    :param tempo_estimates: tempo estimates
    :param beat_references: beat references
    :param tag_references: tag references
    :param output_dir: output directory
    :param size: size
    """

    md.h2('Accuracy')

    md.paragraph(ACC1.description)
    md.paragraph(ACC2.description)
    bib_file_name, cite_key = _extract_bibtex_entry(output_dir, 'Gouyon2006')
    md.paragraph('See [[{cite_key}]({bibtex})].'.format(cite_key=cite_key,
                                                        bibtex=dir_filename(bib_file_name)))

    md.paragraph('Note: When comparing accuracy values for different algorithms, '
                 'keep in mind that an algorithm may have been trained on the '
                 'test set or that the test set may have even been created using '
                 'one of the tested algorithms.',
                 emphasize=True)

    standard_tolerance = 0.04
    if size >= Size.M:
        tolerances = [round(t, 5) for t in np.arange(0.0, 0.0505, 0.0005)]
    else:
        tolerances = [standard_tolerance]

    reference_tempi = ACC1.extract_tempi(tempo_references)
    estimated_tempi = ACC1.extract_tempi(tempo_estimates)

    acc1_results = ACC1.eval_tempi(reference_tempi,
                                   estimated_tempi,
                                   tolerances=tolerances)
    acc2_results = ACC2.eval_tempi(reference_tempi,
                                   estimated_tempi,
                                   tolerances=tolerances)

    # use first entry as point of reference, then compare them using accuracy
    acc1_averages = ACC1.averages(acc1_results)
    acc2_averages = ACC2.averages(acc2_results)

    for ref_name in sorted(acc1_averages.keys()):

        _print_accuracy_table(md, base_name,
                              acc1_results, acc2_results,
                              acc1_averages, acc2_averages,
                              ref_name, tolerances,
                              standard_tolerance,
                              output_dir=output_dir)

        if len(tolerances) > 1:
            _print_tolerance_chart(md, base_name,
                                   ACC1, ref_name,
                                   acc1_averages,
                                   tolerances,
                                   output_dir=output_dir)
            _print_tolerance_chart(md, base_name,
                                   ACC2, ref_name,
                                   acc2_averages,
                                   tolerances,
                                   output_dir=output_dir)

    if size >= Size.L:
        md.h3('Differing Items')
        md.paragraph('For which items did a given estimator not estimate '
                     'a correct value with respect to a given ground truth? '
                     'Are there items which are either very difficult, '
                     'not suitable for the task, or incorrectly annotated and '
                     'therefore never estimated correctly, regardless which '
                     'estimator is used?')
        _print_differing_items(md, base_name, acc1_results,
                               ACC1, tolerances, standard_tolerance,
                               tempo_references, tempo_estimates,
                               beat_references, output_dir=output_dir)

        _print_differing_items(md, base_name, acc2_results,
                               ACC2, tolerances, standard_tolerance,
                               tempo_references, tempo_estimates,
                               beat_references, output_dir=output_dir)

    if len(tempo_estimates) > 1 and size >= Size.M \
            and (ACC1.significant_difference_function is not None or ACC2.significant_difference_function is not None):
        # pick out significance for tolerance = 0.04
        md.h3('Significance of Differences')
        # square table with pvalues
        if ACC1.significant_difference_function is not None:
            _print_significance_matrix(md, base_name, acc1_results,
                                       ACC1, tolerances, standard_tolerance,
                                       output_dir=output_dir)
        if ACC2.significant_difference_function is not None:
            _print_significance_matrix(md, base_name, acc2_results,
                                       ACC2, tolerances, standard_tolerance,
                                       output_dir=output_dir)

    if size >= Size.L:
        _print_subset_evaluations(md, base_name,
                                  [ACC1, ACC2],
                                  None,
                                  tempo_references,
                                  tempo_estimates,
                                  beat_references,
                                  tag_references,
                                  output_dir=output_dir)


def _print_subset_evaluations(md: MarkdownWriter,
                              base_name: str,
                              metrics: List[Metric],
                              eval_results: Optional[List[EvalResults]],
                              tempo_references: Annotations,
                              tempo_estimates: Annotations,
                              beat_references: Annotations,
                              tag_references: TagAnnotations,
                              output_dir: str = './') -> None:
    """
    Print an evaluation over several subsets built around tags,
    tempo intervals or beat stability.

    :param md: markdown writer
    :param base_name: base name for files
    :param metrics: metrics used for reference subsets
    :param eval_results: existing results for the given metrics (same order)
    :param tempo_references: tempo references
    :param tempo_estimates: tempo references
    :param beat_references: beat references
    :param tag_references: tag references
    :param output_dir: output dir
    """

    if tempo_references:

        # pre-compute results for all desired metrics and subset evaluations,
        # unless they are already provided
        if eval_results is None:
            eval_results = [metric.eval_annotations(tempo_references, tempo_estimates)
                            for metric in metrics]

        if beat_references:
            c_var_tresholds = [round(t, 4) for t in np.arange(0, 0.505, 0.005)]
            item_lists = items_lt_c_var(beat_references, c_var_tresholds)
            for metric, result in zip(metrics, eval_results):
                _print_metric_over_c_var(md,
                                         base_name,
                                         metric,
                                         result,
                                         item_lists,
                                         c_var_tresholds,
                                         output_dir=output_dir)

        tol = 10.
        intervals = [(s, s + 2 * tol) for s in np.arange(-tol, 300. - tol + 1.0, 1.0)]
        item_lists = items_in_tempo_intervals(tempo_references, intervals)
        for metric, result in zip(metrics, eval_results):
            _print_metric_over_tempo_interval(md,
                                              base_name,
                                              metric,
                                              result,
                                              item_lists,
                                              intervals,
                                              tol,
                                              output_dir=output_dir)

        for metric, result in zip(metrics, eval_results):
            _print_metric_per_tempo(md,
                                    base_name,
                                    metric,
                                    result,
                                    tempo_references,
                                    output_dir=output_dir)

        if tag_references:
            for metric, result in zip(metrics, eval_results):
                _print_metric_per_tag(md,
                                      base_name,
                                      metric,
                                      result,
                                      tag_references,
                                      output_dir=output_dir)


def _print_metric_over_c_var(md: MarkdownWriter,
                             base_name: str,
                             metric: Metric,
                             eval_results: EvalResults,
                             item_lists: Dict[str, List[Set[str]]],
                             c_var_tresholds: List[float],
                             output_dir: str = './') -> None:
    """
    Prints a subset evaluation over items from the reference
    datasets that fall below a certain coefficient of variation
    (CV).

    :param md: markdown writer
    :param base_name: base name for files
    :param metric: metric
    :param eval_results: eval results
    :param item_lists: pre-computed item lists that correspond
        to a given coefficient of variation threshold
    :param c_var_tresholds: list of used coefficient of variation thresholds
    :param output_dir: output dir
    """

    logging.debug('Printing accuracy over c-var...')

    md.h3('{} on c<sub>var</sub>-Subsets'.format(metric.formatted_name))
    md.paragraph('How well does an estimator perform, when only '
                 'taking tracks into account that have a c<sub>var</sub>-value '
                 'of less than τ, i.e., have a more or less stable beat?')

    legend = list(list(eval_results.values())[0].keys())

    y_axis_label = 'Mean {}'.format(metric.name)
    if metric.unit:
        y_axis_label = 'Mean {} ({})'.format(metric.name, metric.unit)

    for c_var_ref_name in sorted(item_lists.keys()):

        def create_item_filter(threshold_number):
            def filter(ignored, item_id):
                return item_id in item_lists[c_var_ref_name][threshold_number]

            return filter

        item_filters = []
        for threshold_number in range(len(c_var_tresholds)):
            item_filters.append(create_item_filter(threshold_number))

        for tempo_ref_name in sorted(eval_results.keys()):
            y_values = {l: [] for l in legend}
            for threshold_number in range(len(c_var_tresholds)):
                acc = metric.averages({tempo_ref_name: eval_results[tempo_ref_name]},
                                      item_filters[threshold_number],
                                      undefined_value=np.nan)
                reference = acc[tempo_ref_name]
                for l in legend:
                    y_values[l].append(reference[l][0][0])

            values_df = pd.DataFrame(y_values, index=c_var_tresholds)
            values_df.name = 'Mean {} Depending on Coefficient of Variation'.format(metric.name)
            values_df.index.name = 'τ'

            md.h4('{} on c<sub>var</sub>-Subsets for {} based on c<sub>var</sub>-Values from {}'
                  .format(metric.formatted_name,
                          tempo_ref_name,
                          c_var_ref_name))

            _print_line_graph(md, '{}_{}_{}_cv_{}'
                              .format(base_name,
                                       tempo_ref_name,
                                       c_var_ref_name,
                                       metric.name.lower()),
                              values_df,
                              caption='Mean {metric} compared to version {reference} for '
                                      'tracks with c<sub>var</sub> < τ based on beat '
                                      'annotations from {cvar}.'
                              .format(reference=md.to_headline_link(tempo_ref_name),
                                       metric=metric.formatted_name,
                                       cvar=md.to_headline_link(tempo_ref_name)),
                              y_axis_label=y_axis_label,
                              output_dir=output_dir)


def _print_metric_per_tempo(md: MarkdownWriter,
                            base_name: str,
                            metric: Metric,
                            eval_results: EvalResults,
                            tempo_references: Annotations,
                            output_dir: str = './') -> None:
    """
    Prints a GAM-based visualization.

    :param tempo_references: tempo references
    :param md: markdown writer
    :param base_name: base name for files
    :param metric: metric
    :param eval_results: eval results
    :param output_dir: output dir
    """

    logging.debug('Printing estimated {} on tempo...'.format(metric.name))
    md.h3('Estimated {} for Tempo'.format(metric.formatted_name))
    md.paragraph('When fitting a generalized additive model (GAM) to {metric}-values '
                 'and a ground truth, what {metric} can we expect with confidence?'
                 .format(metric=metric.formatted_name))

    legend = list(list(eval_results.values())[0].keys())

    for tempo_ref_name in sorted(eval_results.keys()):

        # get item -> tempo map
        reference_tempi = {item_id: extract_tempo(annotation)
                           for item_id, annotation
                           in tempo_references[tempo_ref_name].items()}

        # ensure an order
        item_order = list(reference_tempi.keys())
        eval_result = eval_results[tempo_ref_name]
        values = {}
        for l in legend:
            r = eval_result[l]
            v = []
            for item_id in item_order:
                if item_id in r:
                    v.append(r[item_id][0])
                else:
                    v.append(np.nan)
            values[l] = v


        index = [reference_tempi[item_order[i]] for i in range(len(item_order))]

        values_df = pd.DataFrame(values, index=index)
        values_df.name = '{} for Tempo in BPM'.format(metric.name)
        values_df.index.name = 'Tempo (BPM)'

        md.h4('Estimated {} for Tempo for {}'
              .format(metric.formatted_name,
                      tempo_ref_name))
        md.paragraph('Predictions of GAMs trained on {metric} '
                     'for estimates for reference {ref}.'
                     .format(ref=md.to_headline_link(tempo_ref_name),
                             metric=metric.formatted_name))

        y_axis_label = '{}'.format(metric.name)
        if metric.unit:
            y_axis_label = '{} ({})'.format(metric.name, metric.unit)

        _print_gam_plot(md, '{}_{}_tempo_gam_{}'
                            .format(base_name,
                                    tempo_ref_name,
                                    metric.name.lower()),
                            values_df,
                            caption='{metric} predictions of a generalized additive model (GAM) '
                                    'fit to {metric} results for {reference}. '
                                    'The 95% confidence interval around the prediction is shaded in gray.'
                            .format(reference=md.to_headline_link(tempo_ref_name),
                                    metric=metric.formatted_name),
                            y_axis_label=y_axis_label,
                            output_dir=output_dir)


def _print_metric_over_tempo_interval(md: MarkdownWriter,
                                      base_name: str,
                                      metric: Metric,
                                      eval_results: EvalResults,
                                      item_lists: Dict[str, List[Set[str]]],
                                      intervals: List[Tuple[float, float]],
                                      tol: float,
                                      output_dir: str = './') -> None:
    """
    Prints a subset evaluation over reference items that fall
    into a specific tempo interval.

    :param md: markdown writer
    :param base_name: base name for files
    :param metric: metric
    :param eval_results: eval results
    :param item_lists: pre-computed item lists that correspond
        to a given tempo interval
    :param intervals: interval
    :param tol: tempo tolerance (width of interval), ``[T-tol,T+tol]`` BPM
    :param output_dir: output dir
    """

    logging.debug('Printing accuracy over interval...')
    md.h3('{} on Tempo-Subsets'.format(metric.formatted_name))

    md.paragraph('How well does an estimator perform, when only '
                 'taking a subset of the reference annotations into '
                 'account? The graphs show mean {} for reference subsets '
                 'with tempi in [T-10,T+10] BPM. '
                 'Note that the graphs do not show confidence '
                 'intervals and that some values may be based on very '
                 'few estimates.'
                 .format(metric.formatted_name))

    interval_centers = [round((i[1]-i[0])/2+i[0], 4) for i in intervals]
    legend = list(list(eval_results.values())[0].keys())

    def create_item_filter(interval_number):
        def filter(tempo_ref_name, item_id):
            return item_id in item_lists[tempo_ref_name][interval_number]
        return filter

    item_filters = []
    for interval_number in range(len(intervals)):
        item_filters.append(create_item_filter(interval_number))

    for tempo_ref_name in sorted(eval_results.keys()):

        y_values = {l: [] for l in legend}

        for interval_number in range(len(intervals)):
            acc = metric.averages({tempo_ref_name: eval_results[tempo_ref_name]},
                                  item_filters[interval_number])
            reference = acc[tempo_ref_name]
            for l in legend:
                y_values[l].append(reference[l][0][0])

        values_df = pd.DataFrame(y_values, index=interval_centers)
        values_df.name = 'Mean {} for Tempo T ±{} BPM'.format(metric.name, tol)
        values_df.index.name = 'T (BPM)'

        md.h4('{} on Tempo-Subsets for {}'
              .format(metric.formatted_name,
                      tempo_ref_name))

        y_axis_label = 'Mean {}'.format(metric.name)
        if metric.unit:
            y_axis_label = 'Mean {} ({})'.format(metric.name, metric.unit)
        _print_line_graph(md,
                           '{}_{}_inter_{}'
                          .format(base_name,
                                   tempo_ref_name,
                                   metric.name.lower()),
                          values_df,
                          caption='Mean {metric} for estimates compared to version {reference} '
                                   'for tempo intervals around T.'
                          .format(reference=md.to_headline_link(tempo_ref_name),
                                   metric=metric.formatted_name),
                          y_axis_label=y_axis_label,
                          output_dir=output_dir)


def _print_metric_per_tag(md: MarkdownWriter,
                          base_name: str,
                          metric: Metric,
                          eval_results: EvalResults,
                          tag_references: TagAnnotations,
                          output_dir: str = './') -> None:
    """
    Prints a subset evaluation over reference items that are
    tagged with a certain tag from one of the supported
    tag namespaces.

    :param md: markdown writer
    :param base_name: base name for files
    :param metric: metric
    :param eval_results: eval results
    :param tag_references: tag reference annotations
    :param output_dir: output directory
    """

    logging.debug('Printing metric over tag...')
    legend = list(list(eval_results.values())[0].keys())

    for namespace, refs in tag_references.items():
        logging.debug('Using namespace \'{}\'.'.format(namespace))

        md.h3('{} for \'{}\' Tags'
              .format(metric.formatted_name, namespace))

        md.paragraph('How well does an estimator perform, '
                     'when only taking tracks into account that are '
                     'tagged with some kind of label? '
                     'Note that some values may be based on very '
                     'few estimates.')

        il = items_per_tag(refs)
        for tag_ref_name, tag_items_dictionary in il.items():

            if tag_items_dictionary:

                for tempo_ref_name in sorted(eval_results.keys()):

                    md.h4('{} for \'{}\' Tags for {}'
                          .format(metric.formatted_name, namespace, tempo_ref_name))

                    if metric != ACC1 \
                            and metric != ACC2 \
                            and metric != PSCORE \
                            and metric != BOTH_CORRECT \
                            and metric != ONE_CORRECT:
                        # if we are not using ACC1 or ACC2, let's do violin plots,
                        # i.e., plot the error distribution
                        tag_error_distributions = {}
                        for tag in sorted(tag_items_dictionary.keys(), reverse=True):
                            item_index = list(tag_items_dictionary[tag])
                            item_index.sort()

                            # create dataframe for raw data
                            raw_columns = {}
                            for estimator, estimates in eval_results[tempo_ref_name].items():
                                raw_columns[estimator] = [estimates.setdefault(i, [np.nan])[0] for i in item_index]
                            raw_values_df = pd.DataFrame(raw_columns, index=item_index)
                            # make sure the columns are in alphabetical order
                            raw_values_df = raw_values_df.reindex(sorted(raw_values_df.columns), axis=1)
                            raw_values_df.index.name = 'Tag'
                            raw_values_df.name = '{} Distributions for Tracks Tagged with \'{}\' Tags'\
                                .format(metric.name, namespace)

                            tag_error_distributions[tag] = raw_values_df
                        #
                        axis_label = '{} Distribution'.format(metric.name)
                        if metric.unit:
                            axis_label = '{} Distribution ({})'.format(metric.name, metric.unit)

                        _print_tag_violin_plot(md,
                                               '{}_{}_{}_{}_{}'
                                               .format(base_name,
                                                       tag_ref_name,
                                                       namespace,
                                                       tempo_ref_name,
                                                       metric.name.lower()),
                                               tag_error_distributions,
                                               caption='{} of estimates compared to version {} '
                                                       'depending on tag '
                                                       'from namespace \'{}\'.'
                                               .format(metric.formatted_name,
                                                       md.to_headline_link(tempo_ref_name),
                                                       namespace),
                                               axis_label=axis_label,
                                               output_dir=output_dir)
                    else:
                        # sadly this is not an error value, so we don't plot a distribution,
                        # but a mean value
                        tags = []
                        tag_accuracies = {l: [] for l in legend}
                        tag_stdev = {l: [] for l in legend}

                        for tag in sorted(tag_items_dictionary.keys(), reverse=True):

                            item_list = tag_items_dictionary[tag]
                            tags.append(tag)
                            acc = metric.averages({tempo_ref_name: eval_results[tempo_ref_name]},
                                                  lambda tempo_ref_name, item_id: item_id in item_list)
                            reference = acc[tempo_ref_name]

                            for l in legend:
                                tag_accuracies[l].append(reference[l][0][0])
                                tag_stdev[l].append(reference[l][1][0])

                        if metric != APE1 and metric != APE2 and metric != AOE1 and metric != AOE2:
                            tag_stdev = None

                        tag_accuracies_df = pd.DataFrame \
                            .from_dict(tag_accuracies,
                                       orient='index',
                                       columns=tags)
                        tag_accuracies_df.name = 'Mean {} for Tracks Tagged with \'{}\' Tags' \
                            .format(metric.name, namespace)
                        tag_accuracies_df.index.name = 'Estimator'
                        tag_accuracies_df.sort_index(inplace=True)
                        tag_stdev_df = None

                        if tag_stdev:
                            tag_stdev_df = pd.DataFrame.from_dict(tag_stdev, orient='index',
                                                                  columns=[l + '_stdev' for l in tags])
                            tag_stdev_df.index.name = 'Estimator'
                            tag_stdev_df.sort_index(inplace=True)

                        x_axis_label = 'Mean {}'.format(metric.name)
                        if metric.unit:
                            x_axis_label = 'Mean {} ({})'.format(metric.name, metric.unit)

                        _print_horizontal_bar_chart(md,
                                                    '{}_{}_{}_{}_{}'
                                                    .format(base_name,
                                                            tag_ref_name,
                                                            namespace,
                                                            tempo_ref_name,
                                                            metric.name.lower()),
                                                    tag_accuracies_df,
                                                    stdev_df=tag_stdev_df,
                                                    caption='Mean {} of estimates compared to version {} '
                                                            'depending on tag '
                                                            'from namespace \'{}\'.'
                                                    .format(metric.formatted_name,
                                                            md.to_headline_link(tempo_ref_name),
                                                            namespace),
                                                    x_axis_label=x_axis_label,
                                                    output_dir=output_dir)


def _print_percentage_for_tag(md: MarkdownWriter,
                              base_name: str,
                              tag_references: TagAnnotations,
                              output_dir: str = './') -> None:
    """
    Print information about what percentage of items/tracks
    are tagged with a given tag.

    :param md: markdown writer
    :param base_name:  base name for files
    :param tag_references: tag reference annotations
    :param output_dir: output dir
    """

    logging.debug('Printing percentage for tag...')

    for namespace in sorted(tag_references.keys()):

        logging.debug('Using namespace \'{}\'.'.format(namespace))
        md.h2('Tag Distribution for \'{}\''.format(namespace))

        namespace_tag_references = tag_references[namespace]

        for version in sorted(namespace_tag_references.keys()):

            tag_counts = {}
            for item_id, annotation in namespace_tag_references[version].items():
                tags = extract_tags(annotation)
                for tag in tags:
                    if tag in tag_counts:
                        tag_counts[tag] += 1
                    else:
                        tag_counts[tag] = 1
            size = len(namespace_tag_references[version])
            tag_counts = {k: [v / size] for k, v in tag_counts.items()}
            tag_counts = OrderedDict(sorted(tag_counts.items(), key=lambda x: x[0]))

            tag_counts_df = pd.DataFrame.from_dict(tag_counts, orient='index', columns=['Count'])
            tag_counts_df.name = 'Percentage of Tracks Tagged With \'{}\' Tags'.format(namespace)
            tag_counts_df.index.name = 'Tag'

            _print_horizontal_bar_chart(md, '{}_{}_{}'
                                        .format(base_name,
                                                version,
                                                namespace),
                                        tag_counts_df,
                                        caption='Percentage of tracks tagged with '
                                                'tags from namespace \'{}\'. '
                                                'Annotations are from reference {}.'
                                        .format(namespace, md.to_headline_link(version)),
                                        x_axis_label='Tracks (%)',
                                        output_dir=output_dir)


def _print_percentage_over_interval(md: MarkdownWriter,
                                    base_name: str,
                                    tempo_annotations: Annotations,
                                    output_dir: str = './') -> None:

    """
    Print smoothed tempo distribution.

    :param md: markdown writer
    :param base_name: base name for files
    :param tempo_annotations: tempo annotations
    :param output_dir: output dir
    """

    logging.debug('Printing percent over interval...')

    tol = 5.0
    step = 1.0
    intervals = [(s, s+2*tol) for s in np.arange(-tol, 300.-tol+step, step)]
    interval_centers = [round((i[1]-i[0])/2+i[0], 4) for i in intervals]
    il = items_in_tempo_intervals(tempo_annotations, intervals)

    legend = list(tempo_annotations.keys())
    y_values = {l: [] for l in legend}
    i = 0

    for version, item_lists in il.items():

        count = float(len(tempo_annotations[version]))
        for item_list in item_lists:
            percentage = len(item_list)/count
            y_values[legend[i]].append(percentage)
        i += 1

    md.h2('Smoothed Tempo Distribution')

    values_df = pd.DataFrame(y_values, index=interval_centers)
    values_df.name = 'Percentage of Annotations for T ±{} BPM'.format(tol)
    values_df.index.name = 'T (BPM)'

    _print_line_graph(md, '{}_dist'.format(base_name),
                      values_df,
                      caption='Percentage of values in tempo interval.',
                      y_axis_label='Values (%)',
                      output_dir=output_dir)

    logging.debug('Printing percent over interval... Done.')


def _alphanumeric_split(tempo_annotations: Annotations) -> Tuple[Annotations, Annotations]:
    """
    Split annotations into two parts (alphanumerically).

    :param tempo_annotations: annotations
    :return: first annotation, other annotations
    """
    sorted_keys = sorted(tempo_annotations.keys())
    first_version = sorted_keys[0]
    first = {first_version: tempo_annotations[first_version]}
    later = {v: tempo_annotations[v] for v in sorted_keys[1:]}
    return first, later


def _highest_mma_split(tempo_estimates: Annotations) -> Tuple[Annotations, Annotations]:
    """
    Split annotations into two parts.
    The first part contains the annotations with
    the highest mean mutual agreement (MMA) based
    on accuracy 1.

    :param tempo_estimates: annotations
    :return: annotations with highest MMA, other annotations
    """
    tempi = ACC1.extract_tempi(tempo_estimates)
    acc = ACC1.averages(ACC1.eval_tempi(tempi, tempi))
    estimate_names = tempo_estimates.keys()
    avg_acc = {}
    for est0 in estimate_names:
        slash0 = est0.find('/')
        s = 0.
        c = 0
        for est1 in estimate_names:
            if est0 == est1:
                continue
            if slash0 != -1:
                slash1 = est1.find('/')
                if slash0 == slash1 and est0[:slash0] == est1[:slash0]:
                    logging.debug('MMA: Detected similar algorithm based on name similarity. '
                                  'Ignoring agreement of \'{est0}\' with \'{est1}\'.'
                                  .format(est0=est0, est1=est1))
                    continue
            c += 1
            s += acc[est0][est1][0][0]
        avg_acc[est0] = s / c
    mma_sorted = sorted(avg_acc.items(), key=operator.itemgetter(1), reverse=True)
    logging.debug('MMA: {}'.format(mma_sorted))
    reference_name = mma_sorted[0][0]
    logging.info('MMA: Highest mean mutual agreement (MMA): {} ({})'
                 .format(reference_name, mma_sorted[0][1]))

    tempo_references = {k: v for k, v in tempo_estimates.items() if k == reference_name}
    tempo_estimates = {k: v for k, v in tempo_estimates.items() if k != reference_name}
    return tempo_references, tempo_estimates


def _print_differing_items(md: MarkdownWriter,
                           base_name: str,
                           comparison_results: EvalResults,
                           metric: Metric,
                           tolerances: List[float],
                           tolerance: float,
                           tempo_references: Annotations,
                           tempo_estimates: Annotations,
                           beat_references: Annotations,
                           output_dir: str = './') -> None:
    """
    Print lists of items that have been annotated differently by
    either different estimators or references.

    :param md: markdown writer
    :param base_name:  base name for files
    :param comparison_results: results
    :param metric: (binary) metric used for the comparison
    :param tolerances: tolerances used during the comparison
    :param tolerance: tolerance we are actually interested in
    :param tempo_references: references
    :param tempo_estimates: estimates
    :param beat_references: beat references
    :param output_dir: output dir
    """

    logging.debug('Printing differing items...')

    tolerance_04_index = tolerances.index(tolerance)
    md.h4('Differing Items {}'.format(metric.formatted_name))
    md.paragraph('Items with different tempo annotations '
                 '({}, 4% tolerance) in different versions:'
                 .format(metric.formatted_name))
    diff = item_ids_for_differing_annotations(comparison_results)

    def tempo_or_nan(values, key):
        if key in values:
            value = extract_tempo(values[key])
        else:
            value = np.nan
        return value

    # find different files between first and others
    diff_in_all = None
    for groundtruth in sorted(diff.keys()):
        for estimate in sorted(diff[groundtruth].keys()):

            differing_items_id = diff[groundtruth][estimate][tolerance_04_index]

            # export to CSV
            csv_file_name = create_file(output_dir,
                                        '{}_{}_{}_diff_items_tol04_{}'
                                        .format(base_name, groundtruth,
                                                estimate,
                                                metric.name.lower()))
            with open(csv_file_name, 'w', newline='', encoding='UTF-8') as csv_file:
                # write csv header
                w = csv.writer(csv_file)
                w.writerow(['Item', 'Reference', 'Estimate'])
                for item_id in sorted(differing_items_id):
                    ref_value = tempo_or_nan(tempo_references[groundtruth], item_id)
                    est_value = tempo_or_nan(tempo_estimates[estimate], item_id)
                    w.writerow([item_id, ref_value, est_value])

            # write markdown
            if differing_items_id:
                md.writeln('{} compared with {} ({} differences):'
                           .format(md.to_headline_link(groundtruth),
                                   md.to_headline_link(estimate),
                                   len(differing_items_id)),
                           emphasize=True)

                # never print more than 10 items...
                for i, item_id in enumerate(sorted(differing_items_id)):
                    md.write('\'{}\' '.format(item_id
                                              .replace('_', '\\_')
                                              .replace('.jams', '')))
                    if i == 10:
                        md.write('...')
                        break

                md.writeln()
                md.paragraph('[CSV]({} "Download list as CSV")'
                             .format(dir_filename(csv_file_name)))
            else:
                md.paragraph('{} compared with {}: No differences.'
                             .format(md.to_headline_link(groundtruth),
                                     md.to_headline_link(estimate)),
                             emphasize=True)

            if diff_in_all is None:
                diff_in_all = set(differing_items_id)
            else:
                diff_in_all = diff_in_all.intersection(diff_in_all, set(differing_items_id))

    if diff_in_all is not None:
        cvs = extract_c_var_from_beats(beat_references)
        first_cvs = {}
        if len(cvs) > 0:
            # simply take the first one for now
            first_cvs = cvs[list(cvs.keys())[0]]

        if len(diff_in_all) > 0:

            # export to CSV
            csv_file_name = create_file(output_dir,
                                        '{}_all_diff_items_tol04_{}'
                                        .format(base_name,
                                                metric.name.lower()))

            references = list(tempo_references.keys())
            estimates = list(tempo_estimates.keys())

            with open(csv_file_name, 'w', newline='', encoding='UTF-8') as csv_file:

                # write csv header
                w = csv.writer(csv_file)
                header = ['Item', 'cvar']
                header.extend(references)
                header.extend(estimates)
                w.writerow(header)
                for item_id in sorted(diff_in_all):
                    values = [item_id]
                    if item_id in first_cvs:
                        values.append(first_cvs[item_id])
                    else:
                        values.append('')

                    values.extend([tempo_or_nan(tempo_references[version], item_id)
                                   for version in references])
                    values.extend([tempo_or_nan(tempo_estimates[version], item_id)
                                   for version in estimates])

                    w.writerow(values)

            # TODO: If tag_open annotations are available, annotate with them
            if len(diff_in_all) == 1:
                md.writeln('None of the estimators estimated the '
                           'following item \'correctly\' using {metric}:'
                           .format(count=len(diff_in_all), metric=metric.formatted_name), strong=True)
            else:
                md.writeln('None of the estimators estimated the '
                           'following {count} items \'correctly\' using {metric}:'
                           .format(count=len(diff_in_all), metric=metric.formatted_name), strong=True)
            found_cvs = []
            for i, item_id in enumerate(sorted(diff_in_all)):
                # never print more than 10 of these
                md.write('\'{}\' '.format(item_id.replace('_', '\\_').replace('.jams', '')))
                if i == 10:
                    md.write('...')
                    break

            if found_cvs:
                if len(found_cvs) > 1:
                    std = stdev(found_cvs)
                else:
                    std = np.nan
                md.write('\n\nAverage c<sub>var</sub>={:.3f}, Stdev={:.3f}'
                         .format(mean(found_cvs),
                                 std))
                md.write('\n\nDataset average c<sub>var</sub>={:.3f}, Stdev={:.3f}'
                         .format(mean(first_cvs.values()),
                                 stdev(first_cvs.values())))

            md.writeln()
            md.paragraph('[CSV]({} "Download list as CSV")'
                         .format(dir_filename(csv_file_name)))
        else:
            md.writeln('All tracks were estimated \'correctly\' '
                       'by at least one system.', strong=True)


def _print_significance_matrix(md: MarkdownWriter,
                               base_name: str,
                               results: EvalResults,
                               metric: Metric,
                               tolerances: List[float] = [0.],
                               tolerance: float = 0.,
                               output_dir: str = './',
                               data_formats: List[str] = ['csv', 'json', 'latex', 'pickle']) -> None:
    """
    Print significance matrix (t-test or McNemar p-values)
    for comparison results.

    .. seealso:: Fabien Gouyon, Anssi P. Klapuri, Simon Dixon, Miguel Alonso,
        George Tzanetakis, Christian Uhle, and Pedro Cano. `An experimental
        comparison of audio tempo induction algorithms.
        <https://www.researchgate.net/profile/Fabien_Gouyon/publication/3457642_An_experimental_comparison_of_audio_tempo_induction_algorithms/links/0fcfd50d982025360f000000/An-experimental-comparison-of-audio-tempo-induction-algorithms.pdf>`_
        IEEE Transactions on Audio, Speech, and Language Processing,
        14(5):1832– 1844, 2006.

    :param md: markdown writer
    :param base_name: base name for files
    :param results:  eval results
    :param metric: metric
    :param tolerances: tolerances used for the results
    :param tolerance: tolerance we are actually interested in
    :param output_dir: output dir
    :param data_formats: data formats to write
    """

    logging.debug('Printing significance difference matrix...')
    tolerance_04_index = tolerances.index(tolerance)
    bib_file_name, cite_key = _extract_bibtex_entry(output_dir, 'Gouyon2006')
    p_values = significant_difference(metric, results)

    if metric.significant_difference_function == mcnemar:
        df_name = 'McNemar p-values'
        caption_template = 'McNemar p-values, using reference annotations '\
                           '{early} as groundtruth with {metric} '\
                           '[[{cite}]({bibtex})]. '\
                           'H<sub>0</sub>: both '\
                           'estimators disagree with the groundtruth to '\
                           'the same amount. If p<=ɑ, reject '\
                           'H<sub>0</sub>, i.e. we have a significant '\
                           'difference in the disagreement with '\
                           'the groundtruth. In the table, p-values<0.05 '\
                           'are set in bold.'
    else:
        df_name = 't-test p-values'
        caption_template = 'Paired t-test p-values, using reference annotations '\
                           '{early} as groundtruth with {metric}. '\
                           'H<sub>0</sub>: ' \
                           'the true mean difference between paired ' \
                           'samples is zero. If p<=ɑ, reject '\
                           'H<sub>0</sub>, i.e. we have a significant '\
                           'difference between estimates from the two algorithms. ' \
                           'In the table, p-values<0.05 are set in bold.'

    for groundtruth in p_values.keys():

        legend = sorted(list(p_values[groundtruth].keys()))
        matrix = {}
        for l in legend:
            matrix[l] = []
            a = p_values[groundtruth][l]
            for j in range(len(legend)):
                pvalue = a[legend[j]][tolerance_04_index]
                matrix[l].append(pvalue)

        values_df = pd.DataFrame(matrix, index=legend)
        values_df.name = df_name
        values_df.index.name = 'Estimator'
        tables = export_data('{}_{}_significance'
                             .format(base_name, metric.name.lower()),
                             values_df,
                             output_dir,
                             data_formats)

        md.table(values_df,
                 caption=caption_template
                 .format(early=md.to_headline_link(groundtruth),
                         metric=metric.formatted_name,
                         cite=cite_key,
                         bibtex=dir_filename(bib_file_name)),
                 strong=lambda p: p <= 0.05,
                 tables=tables)


def _print_accuracy_table(md: MarkdownWriter,
                          name: str,
                          acc1_eval_results: EvalResults,
                          acc2_eval_results: EvalResults,
                          acc1_averages: AverageResults,
                          acc2_averages: AverageResults,
                          ref_name: str,
                          tolerances: List[float],
                          tolerance: float,
                          output_dir: str = './',
                          data_formats: List[str] = ['csv', 'json', 'latex', 'pickle']) -> None:

    """
    Print accuracy results as table.

    :param md: markdown writer
    :param name: name
    :param acc1_eval_results: acc1 (raw) results
    :param acc2_eval_results: acc2 (raw) results
    :param acc1_averages: acc1 (averaged) results
    :param acc2_averages: acc2 (averaged) results
    :param ref_name: dataset name
    :param tolerances: tolerances used
    :param tolerance: tolerance we are interested in
    :param output_dir: output dir
    :param data_formats: desired dataformats
    """

    logging.debug('Printing accuracy table...')
    md.h3('Accuracy Results for {}'.format(ref_name))
    tolerance_04_index = tolerances.index(tolerance)

    # create dataframe for already averaged data
    index = list(acc1_averages[ref_name].keys())
    values_df = pd.DataFrame({
        ACC1.name: [acc1_averages[ref_name][i][0][tolerance_04_index] for i in index],
        ACC2.name: [acc2_averages[ref_name][i][0][tolerance_04_index] for i in index],
    }, index=index)

    values_df.index.name = 'Estimator'
    values_df.sort_values(by=[ACC1.name],
                          ascending=False,
                          inplace=True)
    tables = export_data('{}_{}_accuracy_tol04'.format(name, ref_name),
                         values_df,
                         output_dir,
                         data_formats)

    md.table(values_df,
             column_strong=pd.Series.max,
             caption='Mean accuracy of estimates compared to version {} '
                     'with 4% tolerance ordered by {}.'
             .format(md.to_headline_link(ref_name), ACC1.formatted_name),
             tables=tables)

    def raw(results, metric):
        # create sorted item index
        item_index = set()
        for v in results[ref_name].values():
            item_index = item_index.union(v.keys())
        item_index = list(item_index)
        item_index.sort()
        # create dataframe for raw data
        raw_columns = {}
        for estimator, estimates in results[ref_name].items():
            raw_columns[estimator] = [estimates[i][tolerance_04_index] for i in item_index]
        raw_values_df = pd.DataFrame(raw_columns, index=item_index)
        raw_values_df.index.name = 'Item'

        raw_tables = export_data('{}_{}_raw_{}_tol04'
                                 .format(name, ref_name, metric.name.lower()),
                                 raw_values_df,
                                 output_dir,
                                 data_formats)

        md.write('Raw data {metric}: '.format(metric=metric.formatted_name))
        for k, v in raw_tables.items():
            md.write('[{format}]({path} "Download raw data as {format}") '
                     .format(format=k.upper(), path=v))
        md.paragraph()

    raw(acc1_eval_results, ACC1)
    raw(acc2_eval_results, ACC2)


def _print_mirex_table(md: MarkdownWriter,
                       name: str,
                       metrics: List[Metric],
                       results: List[EvalResults],
                       averages: List[AverageResults],
                       ref_name, tolerances, tolerance, output_dir='./',
                       data_formats: List[str] = ['csv', 'json', 'latex', 'pickle']) -> None:
    """
    Print mirex table.

    :param md: markdown writer
    :param name: name
    :param metrics: metrics used
    :param results: raw results produced by those metrics
    :param averages: average results produced by those metrics
    :param ref_name: dataset name
    :param tolerances: tolerances used
    :param tolerance: tolerance we are interested in
    :param output_dir: output dir
    :param data_formats: desired data formats
    """

    logging.debug('Printing tolerance table...')
    md.h3('MIREX Results for {}'.format(ref_name))
    tolerance_index = tolerances.index(tolerance)

    index = list(averages[0][ref_name].keys())
    values = {metric.name: [result[ref_name][i][0][tolerance_index]
                            for i in index]
              for metric, result in zip(metrics, averages)}
    values_df = pd.DataFrame(values, index=index)

    values_df.index.name = 'Estimator'
    values_df.sort_values(by=[PSCORE.name],
                          ascending=False,
                          inplace=True)
    tables = export_data('{}_{}_tolerance'
                         .format(name, ref_name),
                         values_df,
                         output_dir,
                         data_formats)

    md.table(values_df,
             column_strong=pd.Series.max,
             caption='Compared to {} with {:.1%} tolerance.'
             .format(md.to_headline_link(ref_name), tolerance)
             .format(md.to_headline_link(ref_name)),
             tables=tables)

    def raw(results, metric):
        # create sorted item index
        item_index = set()
        for v in results[ref_name].values():
            item_index = item_index.union(v.keys())
        item_index = list(item_index)
        item_index.sort()
        # create dataframe for raw data
        raw_columns = {}
        for estimator, estimates in results[ref_name].items():
            raw_columns[estimator] = [estimates[i][0] for i in item_index]
        raw_values_df = pd.DataFrame(raw_columns, index=item_index)
        raw_values_df.index.name = 'Item'

        raw_tables = export_data('{}_{}_raw_{}'
                                 .format(name, ref_name, metric.name.lower()),
                                 raw_values_df,
                                 output_dir,
                                 data_formats)

        md.write('Raw data {metric}: '.format(metric=metric.formatted_name))
        for k, v in raw_tables.items():
            md.write('[{format}]({path} "Download raw data as {format}") '
                     .format(metric=metric.formatted_name, format=k.upper(), path=v))
        md.paragraph()

    for i in range(len(metrics)):
        raw(results[i], metrics[i])


def _print_mean_stdev_table(md: MarkdownWriter,
                            name: str,
                            metric1: Metric,
                            metric2: Metric,
                            results1: EvalResults,
                            results2: EvalResults,
                            average_results1: AverageResults,
                            average_results2: AverageResults,
                            ref_name: str,
                            output_dir: str = './',
                            data_formats: List[str] = ['csv', 'json', 'latex', 'pickle']) -> None:
    """
    Print mean/stdev results table.

    :param metric1: ape1 or pe1 or...
    :param metric2: ape2 or pe2 or...
    :param md: markdown writer
    :param name: name
    :param average_results1: mean results for metric1
    :param average_results2: mean results for metric2
    :param ref_name: dataset name
    :param output_dir: output dir
    :param data_formats: desired dataformats
    """
    logging.debug('Printing M{} table...'.format(metric1.name))
    md.h3('Mean {}/{} Results for {}'.format(metric1.formatted_name, metric2.formatted_name, ref_name))

    scale_factor = 1.
    if metric1.unit and metric1.unit == '%':
        scale_factor = 100.

    index = list(average_results1[ref_name].keys())
    values_df = pd.DataFrame({
        metric1.name + '_MEAN': [average_results1[ref_name][name][0][0] for name in index],
        metric1.name + '_STDEV': [average_results1[ref_name][name][1][0] for name in index],
        metric2.name + '_MEAN': [average_results2[ref_name][name][0][0] for name in index],
        metric2.name + '_STDEV': [average_results2[ref_name][name][1][0] for name in index],
    }, index=index).applymap(lambda x: x * scale_factor)  # make sure we have percent values!

    values_df.index.name = 'Estimator'

    # sort by absolute value of either mean or stdev
    order_column = metric1.name + '_MEAN'
    caption = 'Mean {}/{} for estimates compared to version {} ordered by mean.'\
        .format(metric1.name, metric2.name, md.to_headline_link(ref_name))

    if metric1.signed:
        order_column = metric1.name + '_STDEV'
        caption = 'Mean {}/{} for estimates compared to version {} ordered by standard deviation.'\
            .format(metric1.name, metric2.name, md.to_headline_link(ref_name))

    values_df['order'] = values_df[order_column].apply(abs)
    values_df.sort_values(by=['order'], ascending=True, inplace=True)
    del values_df['order']

    tables = export_data('{}_{}_m{}'
                         .format(name, ref_name, metric1.name.lower()),
                         values_df,
                         output_dir,
                         data_formats)

    def closest_to_zero(series):
        """
        Values closest to zero.
        """
        am = series.abs().min()
        if am in series.values:
            return am
        else:
            return -am

    md.table(values_df,
             column_strong=closest_to_zero,
             caption=caption,
             tables=tables)

    def raw(results, metric):
        # create sorted item index
        item_index = set()
        for v in results[ref_name].values():
            item_index = item_index.union(v.keys())
        item_index = list(item_index)
        item_index.sort()
        # create dataframe for raw data
        raw_columns = {}
        for estimator, estimates in results[ref_name].items():
            raw_columns[estimator] = [estimates[i][0] for i in item_index]
        raw_values_df = pd.DataFrame(raw_columns, index=item_index)
        raw_values_df.index.name = 'Item'

        raw_tables = export_data('{}_{}_raw_{}'
                                 .format(name, ref_name, metric.name.lower()),
                                 raw_values_df,
                                 output_dir,
                                 data_formats)

        md.write('Raw data {metric}: '.format(metric=metric.formatted_name))
        for k, v in raw_tables.items():
            md.write('[{format}]({path} "Download raw data as {format}") '
                     .format(format=k.upper(), path=v))
        md.paragraph()

    raw(results1, metric1)
    raw(results2, metric2)


def _print_tolerance_chart(md: MarkdownWriter,
                           base_name: str,
                           metric: Metric,
                           ref_name: str,
                           values: AverageResults,
                           tolerances: Iterable[float],
                           output_dir: str = './') -> None:
    """
    Print tolerance chart.

    :param md: markdown writer
    :param base_name:  base name
    :param metric: metric
    :param ref_name: reference dataset name
    :param values: results
    :param tolerances: tolerances used
    :param output_dir: output dir
    """

    values = values[ref_name]
    md.h3('{} for {}'.format(metric.formatted_name, ref_name))
    legend = sorted(list(values.keys()))
    y_values = {}
    for l in legend:
        y_values[l] = values[l][0]

    if metric.unit:
        metric_label = 'Mean {} ({})'.format(metric.name, metric.unit)
    else:
        metric_label = 'Mean {}'.format(metric.name)

    values_df = pd.DataFrame(y_values, index=tolerances)
    values_df.name = 'Mean {metric} Depending on Tolerance'\
        .format(metric=metric.name)
    values_df.index.name = 'Tolerance (%)'

    _print_line_graph(md,
                       '{}_{}_{}'
                      .format(base_name,
                              ref_name,
                              metric.name.lower().replace(' ', '_')),
                      values_df,
                      caption='Mean {metric} for estimates compared to version '
                              '{reference} depending on tolerance.'
                      .format(reference=md.to_headline_link(ref_name),
                               metric=metric.formatted_name),
                      y_axis_label=metric_label,
                      output_dir=output_dir)


def _print_annotation_metadata(md: MarkdownWriter,
                               annotation_set: Annotations,
                               output_dir: str = './') -> None:
    """
    Print basic annotations metadata contained in
    :py:class:`jams.AnnotationMetadata`.

    Note that we attempt to honor ``bibtex`` and ``ref_url``
    entries in the annotator sandbox. If you are interested
    those showing up, please add them to your jams.

    :param md: markdown writer
    :param annotation_set: annotations
    :param output_dir: output dir
    """

    logging.debug('Printing annotation metadata...')

    def md_table_escape(s):
        return s.replace('|', '\\|')\
            .replace('*', '\\*')\
            .replace('_', '\\_')\
            .replace('\n', '<br>')

    for name in sorted(annotation_set.keys()):
        md.h3('{}'.format(name))

        md.writeln('| Attribute | Value |\n| --- | --- |'.format(name))
        annotations = annotation_set[name]
        values = list(annotations.values())

        if values:
            first_annotation = values[0]
            md.writeln('| **Corpus** | {} |'
                       .format(first_annotation.annotation_metadata.corpus))
            md.writeln('| **Version** | {} |'
                       .format(first_annotation.annotation_metadata.version))
            if first_annotation.annotation_metadata.curator['name']\
                    and first_annotation.annotation_metadata.curator['email']:
                md.writeln('| **Curator** | [{name}](mailto:{email}) |'
                           .format(name=first_annotation.annotation_metadata.curator['name'],
                                   email=first_annotation.annotation_metadata.curator['email']))
            elif first_annotation.annotation_metadata.curator['name']:
                md.writeln('| **Curator** | {name} |'
                           .format(name=first_annotation.annotation_metadata.curator['name']))
            elif first_annotation.annotation_metadata.curator['email']:
                md.writeln('| **Curator** | [{email}](mailto:{email}) |'
                           .format(email=first_annotation.annotation_metadata.curator['email']))
            if first_annotation.annotation_metadata.validation:
                md.writeln('| **Validation** | {} |'
                           .format(first_annotation.annotation_metadata.validation))
            if first_annotation.annotation_metadata.data_source:
                md.writeln('| **Data&nbsp;Source** | {} |'
                           .format(first_annotation.annotation_metadata.data_source))
            if first_annotation.annotation_metadata.annotation_tools:
                md.writeln('| **Annotation&nbsp;Tools** | {} |'
                           .format(first_annotation.annotation_metadata.annotation_tools))
            if first_annotation.annotation_metadata.annotation_rules:
                md.writeln('| **Annotation&nbsp;Rules** | {} |'
                           .format(first_annotation.annotation_metadata.annotation_rules))
            for k in first_annotation.annotation_metadata.annotator.keys():
                if k.lower() == 'bibtex':
                    md.write('| **Annotator,&nbsp;{}** |'.format(k))

                    def write_bibtex(bibtex_entry):
                        try:
                            bib_file_name, first_cite_key = _dump_bibtex_entry(output_dir, bibtex_entry)
                            md.write('[{}]({}) |'
                                       .format(md_table_escape(first_cite_key),
                                               dir_filename(bib_file_name)))
                        except Exception as e:
                            logging.error('Failed to parse annotator bibtex field: {}\nbibtex: {}'
                                          .format(str(e), bibtex_entry))
                            # fallback if parsing goes wrong
                            escaped_annotator = md_table_escape(first_annotation.annotation_metadata.annotator[k])
                            md.writeln('{} |'.format(escaped_annotator))

                    bibtex = first_annotation.annotation_metadata.annotator[k]
                    if isinstance(bibtex, str):
                        write_bibtex(bibtex)
                    else:
                        for b in bibtex:
                            write_bibtex(b)
                    md.writeln()

                elif k.lower() == 'ref_url':
                    md.writeln('| **Annotator,&nbsp;{}** | [{}]({}) |'
                               .format(k, md_table_escape(first_annotation.annotation_metadata.annotator[k]),
                                       first_annotation.annotation_metadata.annotator[k]))
                else:
                    md.writeln('| **Annotator,&nbsp;{}** | {} |'
                               .format(k, md_table_escape(first_annotation.annotation_metadata.annotator[k])))

        md.writeln()


def _dump_bibtex_entry(output_dir: str, bibtex: str) -> Tuple[str, str]:
    """
    Dumps a complete bibtex entry to a file named after the first cite key.

    :param output_dir: output dir
    :param bibtex: complete bibtex entry (not just a cite key)
    :return: name of created file, first cite key
    """
    # parse bibtex
    data = pybtex.database.parse_string(bibtex, bib_format='bibtex')
    first_cite_key = list(data.entries.keys())[0]
    # dump to file
    smkdirs(join(output_dir, 'bib'))
    bib_file_name = join(output_dir, 'bib', '{}.bib'.format(first_cite_key))
    data.to_file(bib_file_name, bib_format='bibtex')
    return bib_file_name, first_cite_key


def _extract_bibtex_entry(output_dir: str, cite_key: str) -> Tuple[str, str]:
    """
    Extracts a bibtex entry from the built-in ``references.bib`` file
    based on the given cite key and dumps it into a separate file,
    which can be used in the generated report.
    The file will be named after the cite key (``cite_key.bib``).

    :param output_dir: output dir
    :param cite_key: cite key
    :return: name of created file, first cite key
    :raises ValueError: if the cite key cannot be found
    """
    references_file = pkg_resources.resource_filename('tempo_eval', 'references.bib')
    database = pybtex.database.parse_file(references_file)
    if cite_key in database.entries:
        entry = database.entries[cite_key]
        entry_database = pybtex.database.BibliographyData()
        entry_database.add_entry(cite_key, entry)
        smkdirs(join(output_dir, 'bib'))
        bib_file_name = join(output_dir, 'bib', '{}.bib'.format(cite_key))
        entry_database.to_file(bib_file_name, bib_format='bibtex')
        return bib_file_name, cite_key
    else:
        raise ValueError('Cite key {} does not exist in \'references.bib\''.format(cite_key))


def _print_basic_statistics(md: MarkdownWriter,
                            name: str,
                            annotation_set: Annotations,
                            estimates: bool = False,
                            output_dir: str = './',
                            data_formats: List[str] = ['csv', 'json', 'latex', 'pickle']) -> None:
    """
    Print basic statistics.

    :param md: markdown writer
    :param name: name
    :param annotation_set: annotation set
    :param estimates: boolean flag indicating whether
        these are estimates or references
    :param output_dir: output dir
    :param data_formats: desired dataformats
    """

    logging.debug('Printing basic statistics...')

    md.h2('Basic Statistics')

    basic_statistics_df = basic_statistics(annotation_set, estimates=estimates)
    tables = export_data(name, basic_statistics_df, output_dir, data_formats)

    md.table(basic_statistics_df,
             caption='Basic statistics.',
             tables=tables,
             float_format=':.2f')


def _print_horizontal_bar_chart(md: MarkdownWriter,
                                name: str,
                                values_df: pd.DataFrame,
                                stdev_df: pd.DataFrame = None,
                                caption: Any = None,
                                x_axis_label: str = None,
                                output_dir: str = './',
                                image_formats: List[str] = ['svg', 'pdf', 'png'],
                                data_formats: List[str] = ['csv', 'json', 'latex', 'pickle']) -> None:
    """
    Print a figure featuring horizontal bars with caption etc.

    :param md: markdown writer
    :param name: name
    :param values_df: values dataframe
    :param stdev_df: stdev dataframe
    :param caption: caption
    :param x_axis_label: x axis label
    :param output_dir: output dir
    :param image_formats: list of image formats
    :param data_formats: list of desired data formats
    """

    images = {}
    file_names = []
    for image_format in image_formats:
        file_name = create_file(output_dir, name, format=image_format)
        file_names.append(file_name)
        images[image_format] = dir_filename(file_name)
    render_horizontal_bar_chart(file_names, values_df,
                                stdev_df=stdev_df,
                                caption=caption,
                                x_axis_label=x_axis_label)

    if stdev_df is not None:
        df = pd.merge(values_df, stdev_df, how='left', left_index=True, right_index=True)
    else:
        df = values_df
    tables = export_data(name, df, output_dir, data_formats)
    md.figure(tables=tables, images=images, caption=caption)


def _print_violin_plot(md: MarkdownWriter,
                       name: str,
                       values_df: pd.DataFrame,
                       caption: Any = None,
                       axis_label: str = None,
                       output_dir: str = './',
                       image_formats: List[str] = ['svg', 'pdf', 'png'],
                       data_formats: List[str] = ['csv', 'json', 'latex', 'pickle']) -> None:
    """
    Print a figure featuring a violin plot with caption etc.

    :param md: markdown writer
    :param name: name
    :param values_df: values dataframe
    :param caption: caption
    :param axis_label: axis label
    :param output_dir: output dir
    :param image_formats: list of image formats
    :param data_formats: list of desired data formats
    """

    images = {}
    file_names = []
    for image_format in image_formats:
        file_name = create_file(output_dir, name, format=image_format)
        file_names.append(file_name)
        images[image_format] = dir_filename(file_name)
    render_violin_plot(file_names, values_df,
                       caption=caption,
                       axis_label=axis_label)

    tables = export_data(name, values_df, output_dir, data_formats)
    md.figure(tables=tables, images=images, caption=caption)


def _print_tag_violin_plot(md: MarkdownWriter,
                           name: str,
                           values_df: Dict[str, pd.DataFrame],
                           caption: Any = None,
                           axis_label: str = None,
                           output_dir: str = './',
                           image_formats: List[str] = ['svg', 'pdf', 'png'],
                           data_formats: List[str] = ['csv', 'json', 'latex', 'pickle']) -> None:
    """
    Print a figure featuring tag-specific violin plots with caption etc.

    :param md: markdown writer
    :param name: name
    :param values_df: dictionary of dataframes with tag names as keys
    :param caption: caption
    :param axis_label: axis label
    :param output_dir: output dir
    :param image_formats: list of image formats
    :param data_formats: list of desired data formats
    """

    images = {}
    file_names = []
    for image_format in image_formats:
        file_name = create_file(output_dir, name, format=image_format)
        file_names.append(file_name)
        images[image_format] = dir_filename(file_name)

    render_tag_violin_plot(file_names, values_df,
                           caption=caption,
                           axis_label=axis_label)

    md.figure(images=images, caption=caption)

    # custom write raw download tags
    # for tag, df in values_df.items():
    #     tables = export_data('{}_{}'.format(name, tag), df, output_dir, data_formats)
    #     md.write('Raw Data for \'{}\': '.format(tag))
    #     for k, v in tables.items():
    #         md.write('[{format}]({path} "Download data as {format}") '.format(format=k.upper(), path=v))
    #     md.paragraph()


def _print_line_graph(md: MarkdownWriter,
                      name: str,
                      values_df: pd.DataFrame,
                      caption: Any = None,
                      y_axis_label: str = None,
                      output_dir: str = './',
                      image_formats: List[str] = ['svg', 'pdf', 'png'],
                      data_formats: List[str] = ['csv', 'json', 'latex', 'pickle']) -> None:
    """
    Print a line graph.

    :param md: markdown writer
    :param name: name
    :param values_df: values dataframe
    :param caption: caption
    :param y_axis_label: y axis label
    :param output_dir: output dir
    :param image_formats: desired image formats
    :param data_formats: desired data formats
    """

    images = {}
    try:
        df_name = values_df.name
    except AttributeError:
        df_name = 'No Name'
    # make sure the columns are in alphabetical order
    values_df = values_df.reindex(sorted(values_df.columns), axis=1)
    # dataframe name does not survive reindexing
    values_df.name = df_name

    file_names = []
    for image_format in image_formats:
        file_name = create_file(output_dir, name, format=image_format)
        file_names.append(file_name)
        images[image_format] = dir_filename(file_name)
    render_line_graph(file_names, values_df,
                      caption=caption,
                      y_axis_label=y_axis_label)

    tables = export_data(name, values_df, output_dir, data_formats)
    md.figure(tables=tables, images=images, caption=caption)


def _print_gam_plot(md: MarkdownWriter,
                    name: str,
                    values_df: pd.DataFrame,
                    confidence_interval: float = 0.95,
                    caption: Any = None,
                    y_axis_label: str = None,
                    output_dir: str = './',
                    image_formats: List[str] = ['svg', 'pdf', 'png'],
                    data_formats: List[str] = ['csv', 'json', 'latex', 'pickle']) -> None:
    """
    Generates and prints a generalized additive model (GAM) that models
    the given dataframe.

    :param confidence_interval: confidence interval of the model
    :param md: markdown writer
    :param name: name
    :param values_df: values dataframe
    :param caption: caption
    :param y_axis_label: y axis label
    :param output_dir: output dir
    :param image_formats: desired image formats
    :param data_formats: desired data formats
    """

    images = {}
    try:
        df_name = values_df.name
    except AttributeError:
        df_name = 'No Name'
    # make sure the columns are in alphabetical order
    values_df = values_df.reindex(sorted(values_df.columns), axis=1)
    # dataframe name does not survive reindexing
    values_df.name = df_name

    x_grid = None
    gam_values_df = None

    for column in values_df.columns:
        Xy = np.array([[v, d]
                       for v, d in zip(values_df.index.values, values_df[column].values)
                       if isfinite(d) and isfinite(v)])

        if Xy.shape[0] == 0:
            raise ValueError('No valid evaluation values in \'\'.'.format(column))

        X = Xy[:,0]
        X = np.expand_dims(X, axis=1)
        y = Xy[:,1]

        gam = LinearGAM().gridsearch(X, y)
        if x_grid is None:
            x_grid = gam.generate_X_grid(term=0, n=100)
            gam_values_df = pd.DataFrame(index=x_grid.reshape(x_grid.shape[0]))
            gam_values_df.name = values_df.name
            gam_values_df.index.name = values_df.index.name

        YY = gam.predict(x_grid)

        gam_values_df[column] = YY
        if confidence_interval:
            cf = gam.confidence_intervals(x_grid)
            gam_values_df[column + "_lowerci"] = cf[:,0]
            gam_values_df[column + "_upperci"] = cf[:,1]

    file_names = []
    for image_format in image_formats:
        file_name = create_file(output_dir, name, format=image_format)
        file_names.append(file_name)
        images[image_format] = dir_filename(file_name)
    render_line_graph(file_names, gam_values_df,
                      confidence_interval=True,
                      caption=caption,
                      y_axis_label=y_axis_label)

    tables = export_data(name, gam_values_df, output_dir, data_formats)
    md.figure(tables=tables, images=images, caption=caption)


def _print_generation_date(md: MarkdownWriter, size: Size = Size.L) -> None:
    """
    Print generation date.

    :param md: markdown writer
    """

    md.rule()
    md.writeln('Generated by [tempo_eval](https://github.com/tempoeval/tempo_eval) {} on {}. Size {}.'
               .format(version, time.strftime('%Y-%m-%d %H:%M'), size.name))


def create_file(output_dir: str, name: str, format: str = 'csv') -> str:
    """
    Create a file based on the given name and with the given extension.
    Escapes certain characters which are not suitable for file names like ``/``.
    All necessary folders leading to the file will be created.
    The file itself will *not* be created.

    :param output_dir: output dir
    :param name: name
    :param format: format, e.g. ``csv``
    :return: file name
    """

    if format in ['csv', 'json', 'latex', 'pickle']:
        dir = 'data'
    else:
        dir = 'figures'
    smkdirs(join(output_dir, dir))
    return join(output_dir, dir, '{}.{}'.format(
        name.replace('/', '_')
            .replace('\\', '_')
            .replace(' ', '_')
            .replace(':', '')
            .replace(';', '')
        , format))


def dir_filename(file_name: str) -> str:
    """
    Return a relative file name consisting only of the last directory name and the
    file name itself.

    :param file_name: path
    :return: e.g. 'data/data.csv'
    """
    return join(basename(dirname(file_name)), basename(file_name))


def export_data(name: str,
                values_df: pd.DataFrame,
                output_dir: str,
                formats: List[str]) -> Dict[str, str]:
    """
    Export the given DataFrame to various formats.

    :param name: base file name without extension or path
    :param values_df: dataframe
    :param output_dir: output dir
    :param formats: array of formats, e.g. ``['csv']``
    :return: dict of format names and filenames
    """
    tables = {}

    if not values_df.index.is_unique:
        values_df = values_df.reset_index()

    for data_format in formats:
        file_name = create_file(output_dir, name, format=data_format)

        logging.debug('Exporting tabular data to file {} using format \'{}\'.'
                      .format(file_name, data_format))

        # lookup export function by name
        f = getattr(pd.DataFrame, 'to_' + data_format)
        if data_format == 'csv':
            f(values_df, file_name, encoding='utf-8')
        else:
            f(values_df, file_name)
        tables[data_format] = dir_filename(file_name)
    return tables


def convert_md_to_html(md_file_name: str,
                       template: str = None,
                       delete: bool = True):
    """
    Converts a markdown file to html.

    :param delete: delete source markdown file after conversion
    :param template: HTML skeleton containing the string ``$CONTENT``,
        which is replaced with the rendered HTML,  ``$VERSION``
        is replaced with the tempo_eval version
    :param md_file_name: markdown file name
    :return: html file name
    """
    logging.debug('Converting file {} from markdown to HTML.'.format(md_file_name))
    if template is None:
        template = HTML_TEMPLATE

    with open(md_file_name, mode='r', encoding='utf-8') as r:
        content = r.read()
        extensions = ['toc', 'tables', 'smarty']
        html = mdown.markdown(content, extensions=extensions, output_format='html5')
        document = template.replace('$CONTENT', html).replace('$VERSION', version)
        html_file_name = md_file_name[:-2] + 'html'
        with open(html_file_name, mode='w', encoding='utf-8') as w:
            w.write(document)

    if delete:
        logging.debug('Deleting {}.'.format(md_file_name))
        remove(md_file_name)

    logging.debug('Created HTML file {}.'.format(html_file_name))

    return html_file_name


def _derive_item_id_fn(base_dir: str,
                       include_relative_dir_in_item_id: bool) -> Callable[[str, jams.JAMS], str]:
    """
    Create a function that derives an item id given a base directory.
    The returned function will **always** try to return the identifier
    ``jam.file_metadata.identifiers['tempo_eval_id']``, if it is available.

    :param include_relative_dir_in_item_id: use relpath
    starting with base_dir or just the file name.
    :param base_dir: base dir
    :return: function to compute item id with
    """

    def extract_tempo_eval_id(jam: jams.JAMS) -> Union[None, str]:
        if 'tempo_eval_id' in jam.file_metadata.identifiers:
            return jam.file_metadata.identifiers['tempo_eval_id']
        else:
            return None

    def base(file: str, jam: jams.JAMS) -> str:
        tempo_eval_id = extract_tempo_eval_id(jam)
        if tempo_eval_id:
            return tempo_eval_id
        else:
            return basename(file)

    def rel(file: str, jam: jams.JAMS) -> str:
        tempo_eval_id = extract_tempo_eval_id(jam)
        if tempo_eval_id:
            return tempo_eval_id
        else:
            return relpath(file, base_dir)

    if include_relative_dir_in_item_id and base_dir is not None:
        return rel
    else:
        return base
