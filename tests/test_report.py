import logging
import os
from os.path import join, exists, dirname, basename

import jams
import pytest

from tempo_eval.report import print_corpus_report, print_report, Size


def test_print_report_basic(tmpdir):

    dir = str(tmpdir)

    ref_dir = join(dir, 'ref')
    est_dir = join(dir, 'est')

    os.makedirs(ref_dir)
    os.makedirs(est_dir)

    # create two jam files
    version = '1.0'
    item_id = 'jam0.jams'

    corpus = 'corpus spaces and slash /'
    _create_jam_file(ref_dir, version, item_id, 100, corpus=corpus)
    _create_jam_file(est_dir, version, item_id, 100.1, corpus=corpus)

    # expected artifacts
    expected_data_files = {
        Size.L.name: [
            'all_variation',
            '{corpus}_estimates_{corpus}_1.0_accuracy1',
            '{corpus}_estimates_{corpus}_1.0_accuracy2',
            '{corpus}_estimates_{corpus}_1.0_accuracy_tol04',
            '{corpus}_estimates_{corpus}_1.0_p-score',
            '{corpus}_estimates_{corpus}_1.0_one_correct',
            '{corpus}_estimates_{corpus}_1.0_both_correct',
            '{corpus}_estimates_{corpus}_1.0_maoe1',
            '{corpus}_estimates_{corpus}_1.0_moe1',
            '{corpus}_estimates_{corpus}_1.0_distribution_aoe1',
            '{corpus}_estimates_{corpus}_1.0_distribution_aoe2',
            '{corpus}_estimates_{corpus}_1.0_distribution_oe1',
            '{corpus}_estimates_{corpus}_1.0_distribution_oe2',
            '{corpus}_estimates_{corpus}_1.0_inter_accuracy1',
            '{corpus}_estimates_{corpus}_1.0_inter_accuracy2',
            '{corpus}_estimates_{corpus}_1.0_inter_aoe1',
            '{corpus}_estimates_{corpus}_1.0_inter_aoe2',
            '{corpus}_estimates_{corpus}_1.0_inter_oe1',
            '{corpus}_estimates_{corpus}_1.0_inter_oe2',
            '{corpus}_estimates_{corpus}_1.0_inter_p-score',
            '{corpus}_estimates_{corpus}_1.0_inter_one_correct',
            '{corpus}_estimates_{corpus}_1.0_inter_both_correct',
            '{corpus}_estimates_{corpus}_1.0_tempo_gam_accuracy1',
            '{corpus}_estimates_{corpus}_1.0_tempo_gam_accuracy2',
            '{corpus}_estimates_{corpus}_1.0_tempo_gam_aoe1',
            '{corpus}_estimates_{corpus}_1.0_tempo_gam_aoe2',
            '{corpus}_estimates_{corpus}_1.0_tempo_gam_oe1',
            '{corpus}_estimates_{corpus}_1.0_tempo_gam_oe2',
            '{corpus}_estimates_{corpus}_1.0_tempo_gam_p-score',
            '{corpus}_estimates_{corpus}_1.0_tempo_gam_one_correct',
            '{corpus}_estimates_{corpus}_1.0_tempo_gam_both_correct',
            '{corpus}_estimates_{corpus}_1.0_{corpus}_1.0_diff_items_tol04_accuracy1',
            '{corpus}_estimates_{corpus}_1.0_{corpus}_1.0_diff_items_tol04_accuracy2',
            '{corpus}_estimates_{corpus}_1.0_tolerance',
            '{corpus}_estimates_basic_stats',
            '{corpus}_estimates_dist',
            '{corpus}_reference_basic_stats',
            '{corpus}_reference_dist',
        ],
        Size.M.name: [
            '{corpus}_estimates_{corpus}_1.0_accuracy1',
            '{corpus}_estimates_{corpus}_1.0_accuracy2',
            '{corpus}_estimates_{corpus}_1.0_accuracy_tol04',
            '{corpus}_estimates_{corpus}_1.0_p-score',
            '{corpus}_estimates_{corpus}_1.0_one_correct',
            '{corpus}_estimates_{corpus}_1.0_both_correct',
            '{corpus}_estimates_{corpus}_1.0_maoe1',
            '{corpus}_estimates_{corpus}_1.0_moe1',
            '{corpus}_estimates_{corpus}_1.0_distribution_aoe1',
            '{corpus}_estimates_{corpus}_1.0_distribution_aoe2',
            '{corpus}_estimates_{corpus}_1.0_distribution_oe1',
            '{corpus}_estimates_{corpus}_1.0_distribution_oe2',
            '{corpus}_estimates_{corpus}_1.0_tolerance',
            '{corpus}_estimates_basic_stats',
            '{corpus}_estimates_dist',
            '{corpus}_reference_basic_stats',
            '{corpus}_reference_dist',
        ],
        Size.S.name: [
            '{corpus}_estimates_{corpus}_1.0_accuracy_tol04',
            '{corpus}_estimates_{corpus}_1.0_maoe1',
            '{corpus}_estimates_{corpus}_1.0_moe1',
            '{corpus}_estimates_{corpus}_1.0_tolerance',
            '{corpus}_estimates_{corpus}_1.0_distribution_aoe1',
            '{corpus}_estimates_{corpus}_1.0_distribution_aoe2',
            '{corpus}_estimates_{corpus}_1.0_distribution_oe1',
            '{corpus}_estimates_{corpus}_1.0_distribution_oe2',
            '{corpus}_estimates_basic_stats',
            '{corpus}_reference_basic_stats',
        ]
    }
    for size in Size:
        if size == Size.XL:
            logging.warning('XL output is not tested')
            continue

        out_dir = join(dir, 'out_{}'.format(size.name))

        report = print_report(output_dir=out_dir,
                              estimates_dir=est_dir,
                              references_dir=ref_dir,
                              validate=False,
                              size=size)
        assert exists(report)
        escaped_corpus = corpus.replace(' ', '_').replace('/', '_')
        corpus_report_file = join(dirname(report), escaped_corpus + '.md')
        assert exists(corpus_report_file)
        with open(report, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert basename(corpus_report_file) in content

        for f in expected_data_files[size.name]:
            file = join(dirname(report), 'data', f.format(corpus=escaped_corpus) + '.csv')
            assert exists(file)
        # TODO: other formats? figures?


def test_print_html_report_basic(tmpdir):
    dir = str(tmpdir)

    ref_dir = join(dir, 'ref')
    est_dir = join(dir, 'est')
    out_dir = join(dir, 'out')

    os.makedirs(ref_dir)
    os.makedirs(est_dir)

    # create two jam files
    version = '1.0'
    item_id = 'jam0.jams'

    corpus = 'corpus spaces and slash /'
    _create_jam_file(ref_dir, version, item_id, 100, corpus=corpus)
    _create_jam_file(est_dir, version, item_id, 100.1, corpus=corpus)

    expected_data_files = [
        '{corpus}_estimates_{corpus}_1.0_accuracy_tol04',
        '{corpus}_estimates_{corpus}_1.0_maoe1',
        '{corpus}_estimates_{corpus}_1.0_moe1',
        '{corpus}_estimates_{corpus}_1.0_tolerance',
        '{corpus}_estimates_{corpus}_1.0_distribution_aoe1',
        '{corpus}_estimates_{corpus}_1.0_distribution_aoe2',
        '{corpus}_estimates_{corpus}_1.0_distribution_oe1',
        '{corpus}_estimates_{corpus}_1.0_distribution_oe2',
        '{corpus}_estimates_basic_stats',
        '{corpus}_reference_basic_stats',
    ]
    report = print_report(output_dir=out_dir,
                          estimates_dir=est_dir,
                          references_dir=ref_dir,
                          validate=False,
                          format='html',
                          size=Size.S)

    assert exists(report)
    assert report.endswith('.html')
    escaped_corpus = corpus.replace(' ', '_').replace('/', '_')
    corpus_report_file = join(dirname(report), escaped_corpus + '.html')
    assert exists(corpus_report_file)
    with open(report, mode='r', encoding='utf-8') as r:
        content = r.read()
        assert basename(corpus_report_file) in content

    for f in expected_data_files:
        file = join(dirname(report), 'data', f.format(corpus=escaped_corpus) + '.csv')
        assert exists(file)


def _create_jam_file(dir, version, item_id, bpm, corpus='corpus a'):
    jam = jams.JAMS()
    jam.file_metadata.duration = 100
    tempo = jams.Annotation(namespace='tempo')
    tempo.append(time=0.0,
                 duration='nan',
                 value=bpm,
                 confidence=0.9)
    tempo.append(time=0.0,
                 duration='nan',
                 value=bpm * 2,
                 confidence=0.1)
    tempo.annotation_metadata = jams.AnnotationMetadata(
        corpus=corpus,
        annotation_rules='rules',
        annotation_tools='tools',
        data_source='data source',
        validation='validation method',
        annotator={
            'name': 'annotator name',
            'email': 'email@somewhere.com',
            'bibtex': '@inproceedings{}',
            'ref_url': 'http://www.someurl.com/'
        },
        curator={
            'name': 'curator name',
            'email': 'curator@mir.com',
        },
        version=version)
    jam.annotations.append(tempo)
    # add beats and tags
    jam.save(join(dir, item_id))


def test_print_report(tmpdir):

    output_dir = str(tmpdir)

    print_report(output_dir=output_dir,
                 validate=False,
                 corpus_names=['gtzan'])


def test_print_report_format(tmpdir):

    output_dir = str(tmpdir)

    with pytest.raises(ValueError):
        print_report(output_dir=output_dir,
                     validate=False,
                     corpus_names=['gtzan'],
                     format='nonsense')

    with pytest.raises(ValueError):
        print_report(output_dir=output_dir,
                     validate=False,
                     corpus_names=['gtzan'],
                     format=None)


def test_print_corpus_report(tmpdir):

    output_dir = str(tmpdir)

    print_corpus_report('ballroom',
                        output_dir=output_dir,
                        validate=False,
                        size=Size.S)

    # print_corpus_report('gtzan',
    #                     output_dir=output_dir,
    #                     validate=False)

    # print_corpus_report('giantsteps_tempo',
    #                     output_dir=output_dir,
    #                     validate=False)

    # print_corpus_report('beatles', output_dir=output_dir, validate=False)
    # print_corpus_report('fma_small', output_dir=output_dir, validate=False)
    # print_corpus_report('giantsteps_tempo', output_dir=output_dir, validate=False)
    # print_corpus_report('hainsworth', output_dir=output_dir, validate=False)
    # print_corpus_report('ismir2004songs', output_dir=output_dir, validate=False)
    # print_corpus_report('lmd_tempo', output_dir=output_dir, validate=False)
    # print_corpus_report('smc_mirex', output_dir=output_dir, validate=False)
    # print_corpus_report('wjd', output_dir=output_dir, validate=False)
