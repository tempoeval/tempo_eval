import math
from os.path import join

import jams
import pytest
import numpy as np
from numba import TypingError
from pytest import approx
from math import isnan

from tempo_eval.evaluation import list_reference_corpus_names, read_reference_annotations, read_estimate_annotations, \
    significant_difference, fraction_lt_thresholds, ACC1, is_single_bpm, is_mirex_style, equal1, equal2, \
    p_score, one_correct, both_correct, ape1, ape2, list_estimate_corpus_names, read_annotations, read_reference_tags, \
    item_ids_for_differing_annotations, extract_tempi_and_salience, extract_c_var_from_beats, items_lt_c_var, \
    items_in_tempo_intervals, items_per_tag, basic_statistics, sweet_octave, extract_tempo, pe1, pe2, PE1, APE2, PE2, \
    APE1


def test_list_reference_corpus_names():
    references = list_reference_corpus_names()
    print('Built-in reference datasets: {}'.format(references))
    assert len([e for e in references if e.startswith('.')]) == 0
    assert 'gtzan' in references


def test_list_estimate_corpus_names():
    references = list_estimate_corpus_names()
    print('Built-in estimates: {}'.format(references))
    assert len([e for e in references if e.startswith('.')]) == 0
    assert 'gtzan' in references


def test_read_annotations_empty(tmpdir):

    # minimal test, just make sure nothing bad happens
    dir = str(tmpdir)
    # create two jam files
    jam0 = jams.JAMS()
    jam0.file_metadata.duration = 100
    jam0.save(join(dir, 'jam0.jams'))
    jam1 = jams.JAMS()
    jam1.file_metadata.duration = 200
    jam1.save(join(dir, 'jam1.jams'))

    annotations = read_annotations(dir)
    assert annotations == {}


def test_read_annotations(tmpdir):
    dir = str(tmpdir)

    # create two jam files
    jam0 = jams.JAMS()
    jam0.file_metadata.duration = 100
    tempo0 = jams.Annotation(namespace='tempo')
    tempo0.append(time=0.0,
                  duration='nan',
                  value=100,
                  confidence=0.9)
    tempo0.annotation_metadata = jams.AnnotationMetadata(
        corpus='corpus a', version='1.0'
    )

    jam0.annotations.append(tempo0)
    jam0.save(join(dir, 'jam0.jams'))

    jam1 = jams.JAMS()
    tempo1 = jams.Annotation(namespace='tempo')
    tempo1.append(time=0.0,
                  duration='nan',
                  value=200,
                  confidence=0.8)
    tempo1.annotation_metadata = jams.AnnotationMetadata(
        corpus='corpus a', version='1.0'
    )
    jam1.annotations.append(tempo1)
    jam1.file_metadata.duration = 200
    jam1.save(join(dir, 'jam1.jams'))

    annotations = read_annotations(dir)['tempo']
    assert ['1.0'] == list(annotations.keys())
    assert {'jam0.jams', 'jam1.jams'} == set(annotations['1.0'].keys())
    assert type(annotations['1.0']['jam0.jams']) == jams.Annotation
    assert annotations['1.0']['jam0.jams']['data'][0].value == 100
    assert annotations['1.0']['jam1.jams']['data'][0].value == 200


def test_read_reference_annotations():
    ballroom = read_reference_annotations('ballroom', validate=False)['tempo']
    assert ballroom is not None
    assert '1.0' in ballroom
    annotation = ballroom['1.0']['Albums-Latin_Jam4-12.jams']
    assert annotation is not None


def test_read_reference_tags():
    ballroom_tags = read_reference_tags('ballroom', validate=False)
    assert ballroom_tags is not None
    assert 'tag_open' in ballroom_tags
    annotation = ballroom_tags['tag_open']['1.0']['Albums-Latin_Jam4-12.jams']
    assert annotation['data'][0].value == 'Samba'

    gtzan_tags = read_reference_tags('gtzan', validate=False)
    assert gtzan_tags is not None
    assert 'tag_gtzan' in gtzan_tags
    annotation = gtzan_tags['tag_gtzan']['1.0']['reggae.00010.jams']
    assert annotation['data'][0].value == 'reggae'


def test_estimate_annotations():
    ballroom = read_estimate_annotations('ballroom', validate=False)['tempo']
    assert ballroom is not None
    assert 'schreiber2018/fcn' in ballroom
    annotation = ballroom['schreiber2018/fcn']['Albums-Latin_Jam4-12.jams']
    assert annotation is not None


def test_acc1_basic():
    corpus = 'smc_mirex'
    estimates = read_estimate_annotations(corpus, validate=False)['tempo']
    references = read_reference_annotations(corpus, validate=False)['tempo']
    eval_result = ACC1.eval_annotations(references, estimates)
    acc = ACC1.averages(eval_result)
    assert set(references.keys()) == set(acc.keys())
    assert set(estimates.keys()) == set(acc['1.0'].keys())
    avg = acc['1.0']['schreiber2018/fcn']
    assert len(avg) == 2
    assert len(avg[0]) == 1
    assert len(avg[1]) == 1
    assert 0.0 <= avg[0] <= 1.0
    assert 0.0 <= avg[1] <= 1.0

    assert ACC1.is_tempo_suitable(10.0)
    assert not ACC1.is_tempo_suitable('10.0')


def test_is_single_bpm():
    assert is_single_bpm(5.)
    assert not is_single_bpm('wrong type')
    assert not is_single_bpm((5., 5.))
    assert not is_single_bpm(-5)
    assert not is_single_bpm(10000)


def test_is_mirex_style():
    assert is_mirex_style((5., 6., 0.2))
    assert not is_mirex_style((0, 6., 0.2))
    assert not is_mirex_style((1, 6., 0.))
    assert not is_mirex_style((1, 6., 1.))
    assert not is_mirex_style('wrong type')
    assert not is_mirex_style(1.)
    assert not is_mirex_style((5., 0, 0.2))
    assert not is_mirex_style((5., 6., 1.1))


def test_equal1():
    assert equal1(10., 10., tolerance=0.0)
    assert equal1(10., 10.3999999, tolerance=0.04)
    assert not equal1(10., 10.00000001, tolerance=0.0)
    assert not equal1(10., None)
    # do not show tolerance for wrong types
    with pytest.raises((TypeError, TypingError)):
        assert not equal1('10.', 10., tolerance=0.0)
    with pytest.raises(ValueError):
        # bad tolerance
        equal1(2., 3., tolerance=2.0)


def test_equal2():
    assert equal2(10., 10., tolerance=0.00001)
    assert equal2(10., 20., tolerance=0.00001)
    assert equal2(10., 30., tolerance=0.00001)
    assert equal2(10., 5., tolerance=0.00001)
    assert equal2(10., 10. / 3., tolerance=0.00001)

    assert equal2(10., 10.3999999, tolerance=0.04)
    assert not equal2(10., 10.00000001, tolerance=0.0)
    assert not equal2(10., None)
    # do not show tolerance for wrong types
    with pytest.raises((TypeError, TypingError)):
        assert not equal2('10.', 10., tolerance=0.0)


def test_p_score():
    assert p_score((10., 20., 0.5), (10., 20., 0.5), tolerance=0.0) == 1.
    assert p_score((10., 20., 0.5), (10., 20., 0.75), tolerance=0.0) == 1.
    assert p_score((10., 20., 0.75), (10., 50., 0.75), tolerance=0.0) == 0.75
    assert p_score((10., 20., 0.75), (15., 20., 0.75), tolerance=0.0) == 0.25

    assert p_score((10., 20., 0.5), (10.5, 20.5, 0.5), tolerance=0.10) == 1.
    assert p_score((10., 20., 0.5), (10.5, 21., 0.5), tolerance=0.04) == 0.
    assert p_score((10., 20., 0.5), None) == 0.

    with pytest.raises(ValueError):
        # bad tolerance
        p_score((10., 20., 0.75), (15., 20., 0.75), tolerance=2.0)


def test_one_correct():
    assert one_correct((10., 20., 0.5), (10., 20., 0.5), tolerance=0.0)
    assert one_correct((10., 20., 0.5), (10., 20., 0.75), tolerance=0.0)
    assert one_correct((10., 20., 0.75), (10., 50., 0.75), tolerance=0.0)
    assert one_correct((10., 20., 0.75), (15., 20., 0.75), tolerance=0.0)

    assert one_correct((10., 20., 0.5), (10.5, 20.5, 0.5), tolerance=0.10)
    assert not one_correct((10., 20., 0.5), None)
    assert not one_correct((10., 20., 0.5), (10.5, 21., 0.5), tolerance=0.04)
    assert not one_correct((10., 20., 0.75), (15., 25., 0.75), tolerance=0.0)


def test_both_correct():
    assert both_correct((10., 20., 0.5), (10., 20., 0.5), tolerance=0.0)
    assert both_correct((10., 20., 0.5), (10., 20., 0.75), tolerance=0.0)
    assert not both_correct((10., 20., 0.75), (10., 50., 0.75), tolerance=0.0)
    assert not both_correct((10., 20., 0.75), (15., 20., 0.75), tolerance=0.0)

    assert not both_correct((10, 20., 0.5), None)
    assert both_correct((10., 20., 0.5), (10.5, 20.5, 0.5), tolerance=0.10)
    assert not both_correct((10., 20., 0.5), (10.5, 21., 0.5), tolerance=0.04)
    assert not both_correct((10., 20., 0.75), (15., 25., 0.75), tolerance=0.0)


def test_ape1():
    assert isnan(ape1(10., None))
    assert ape1(10., 10.) == 0.
    assert ape1(10., 10.3999999) == approx(0.03999999)
    assert ape1(10.3999999, 10.) == approx(0.03846152)
    assert ape1(10., 10.00000001) == approx(0.000000001)
    # do not show tolerance for wrong types
    with pytest.raises((TypeError, TypingError)):
        assert not ape1('10.', 10.)
    assert math.isnan(ape1(0, 10.))
    assert not APE1.signed


def test_ape2():
    assert isnan(ape2(10., None))
    assert ape2(10., 10.) == 0.
    assert ape2(10., 20.) == approx(0.)
    assert ape2(10., 30.) == approx(0.)
    assert ape2(10., 5.) == approx(0.)
    assert ape2(10., 10./3.) == approx(0.)
    assert ape2(10., 10.3999999) == approx(0.03999999)
    assert ape2(10.3999999, 10.) == approx(0.03846152)
    assert ape2(10., 10.00000001) == approx(0.000000001)
    # do not show tolerance for wrong types
    with pytest.raises((TypeError, TypingError)):
        assert not ape2('10.', 10.)
    assert math.isnan(ape2(0, 10.))
    assert not APE2.signed


def test_pe1():
    assert isnan(pe1(10., None))
    assert pe1(10., 10.) == 0.
    assert pe1(10., 10.3999999) == approx(0.03999999)
    assert pe1(10.3999999, 10.) == approx(-0.03846152)
    assert pe1(10., 10.00000001) == approx(0.000000001)
    # do not show tolerance for wrong types
    with pytest.raises((TypeError, TypingError)):
        assert not pe1('10.', 10.)
    assert math.isnan(pe1(0, 10.))
    assert PE1.signed


def test_pe2():
    assert isnan(pe2(10., None))
    assert pe2(10., 10.) == 0.
    assert pe2(10., 20.) == approx(0.)
    assert pe2(10., 30.) == approx(0.)
    assert pe2(10., 5.) == approx(0.)
    assert pe2(10., 10./3.) == approx(0.)
    assert pe2(10., 10.3999999) == approx(0.03999999)
    assert pe2(10.3999999, 10.) == approx(-0.03846152)
    assert pe2(10., 10.00000001) == approx(0.000000001)
    # do not show tolerance for wrong types
    with pytest.raises((TypeError, TypingError)):
        assert not pe2('10.', 10.)
    assert math.isnan(pe2(0, 10.))
    assert PE2.signed

    
def test_fraction_below_threshold():
    fraction = fraction_lt_thresholds([1, 4, 2, 3], [4, 3])
    assert fraction == [0.75, 0.5]


def test_item_ids_for_differing_annotations():
    corpus = 'smc_mirex'
    estimates = read_estimate_annotations(corpus, validate=False)['tempo']
    references = read_reference_annotations(corpus, validate=False)['tempo']
    eval_result = ACC1.eval_annotations(references, estimates)
    item_ids = item_ids_for_differing_annotations(eval_result)
    ids = item_ids['1.0']['schreiber2018/fcn'][0]
    assert 'SMC_194.jams' in ids


def test_significant_difference():
    corpus = 'smc_mirex'
    estimates = read_estimate_annotations(corpus, validate=False)['tempo']
    references = read_reference_annotations(corpus, validate=False)['tempo']
    eval_result = ACC1.eval_annotations(references, estimates, tolerances=[0.04, 0.04])
    significances = significant_difference(ACC1, eval_result)
    for reference in significances.keys():
        for estimator1 in significances[reference].keys():
            for estimator2 in significances[reference][estimator1].keys():
                p_values = significances[reference][estimator1][estimator2]
                if estimator1 == estimator2:
                    assert all([p_value == 1.0 for p_value in p_values])
                else:
                    # check symmetry
                    assert significances[reference][estimator2][estimator1] == p_values
                    assert all([p_value >= 0.0 for p_value in p_values])


def test_extract_tempi_and_salience():
    references = read_reference_annotations('giantsteps_tempo', validate=False)['tempo']
    for name, items in references.items():
        for item_id, annotation in items.items():
            tempo = extract_tempi_and_salience(annotation)
            assert len(tempo) == 3
            assert tempo[0] < tempo[1]
            assert 0. <= tempo[2] <= 1.


def test_extract_c_var_from_beats():
    annotations = read_reference_annotations('ballroom', namespace='beat', validate=False)['beat']
    c_vars = extract_c_var_from_beats(annotations)
    assert c_vars['1.0']['Media-105602.jams'] == approx(0.0044346183, rel=1e-2)


def test_items_lt_c_var():

    annotations = read_reference_annotations('ballroom', namespace='beat', validate=False)['beat']

    # default thresholds
    items = items_lt_c_var(annotations)
    # none of the items has c_var 0
    assert set() == items['1.0'][0]
    assert 'Albums-Latin_Jam4-12.jams' in items['1.0'][1]

    # explicit thresholds
    items = items_lt_c_var(annotations, thresholds=[0, 20])
    # none of the items has c_var 0
    assert len(items['1.0'][0]) == 0
    # all items have c_var 0 < 20
    assert len(items['1.0'][1]) == len(annotations['1.0'])


def test_items_in_tempo_intervals():

    annotations = read_reference_annotations('ballroom', validate=False)['tempo']

    # default intervals
    items = items_in_tempo_intervals(annotations)
    # none of the items is in the first interval
    assert set() == items['1.0'][0]
    # Media-105213.jams has tempo 60 in reference ground truth '1.0'
    # it should therefore be in the interval [53, 53+11] BPM:
    assert 'Media-105213.jams' in items['1.0'][53]
    # but not in [61, 61+11] BPM:
    assert 'Media-105213.jams' not in items['1.0'][61]

    # explicit intervals
    items = items_in_tempo_intervals(annotations, intervals=[(0, 10), (0, 300)])
    # none of the items is in the first interval
    assert len(items['1.0'][0]) == 0
    # all items are in [0, 300] BPM
    assert len(items['1.0'][1]) == len(annotations['1.0'])


def test_items_per_tag():
    tags = read_reference_tags('ballroom', validate=False)
    items = items_per_tag(tags['tag_open'])
    assert 'Samba' in items['1.0']
    assert 'Albums-Latin_Jam4-12.jams' in items['1.0']['Samba']


def test_basic_statistics():
    annotations = read_reference_annotations('ballroom', validate=False)['tempo']
    statistics = basic_statistics(annotations)
    assert len(statistics.columns) == 7
    assert statistics.Size.mean() == 698


def test_sweet_octave():
    # test with artificial data
    octave, percentage = sweet_octave([50, 99])
    assert octave == 50
    assert percentage == 1.0

    octave, percentage = sweet_octave([50., 100.])
    assert octave == 26  # lowest possible j -> [26, 51)
    assert percentage == 0.5

    # check with NaN and Inf
    octave, percentage = sweet_octave([50., 100., np.nan, np.inf])
    assert octave == 26  # lowest possible j -> [26, 51)
    assert percentage == 0.25

    # test with ballroom data
    annotations = read_reference_annotations('ballroom', validate=False)['tempo']
    tempi = [extract_tempo(annotation) for annotation in annotations['1.0'].values()]
    octave, percentage = sweet_octave(tempi)
    assert octave == 91
    assert percentage == approx(0.7134670487)