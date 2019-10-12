"""
API Reference
=============

Print Reports
-------------

.. autosummary::
    :toctree: generated/

    print_report
    print_corpus_report


Compare Tempo Values
--------------------

.. autosummary::
    :toctree: generated_functions/

    equal1
    equal2
    ape1
    ape2
    pe1
    pe2
    oe1
    oe2
    aoe1
    aoe2
    p_score
    one_correct
    both_correct


Access Annotations
------------------

.. autosummary::
    :toctree: generated/

    list_estimate_corpus_names
    list_reference_corpus_names
    read_estimate_annotations
    read_reference_annotations
    read_annotations
    read_reference_tags
    get_estimates_path
    get_references_path


Extract Values From JAMS
------------------------

.. autosummary::
    :toctree: generated/

    extract_tempo
    extract_tempi_and_salience
    extract_tempo_from_beats
    extract_c_var_from_beats
    extract_tags
    is_mirex_style
    is_single_bpm

Metrics
--------
.. autosummary::
    :toctree: generated/

    Metric
    ACC1
    ACC2
    APE1
    APE2
    PE1
    PE2
    OE1
    OE2
    AOE1
    AOE2
    PSCORE
    ONE_CORRECT
    BOTH_CORRECT

"""

from .report import print_report, print_corpus_report
from .evaluation import equal1, equal2, ape1, ape2, pe1, pe2, oe1, oe2, aoe1, aoe2, \
    both_correct, one_correct, p_score
from .evaluation import read_reference_tags,\
    read_annotations, read_estimate_annotations, read_reference_annotations, \
    list_estimate_corpus_names, list_reference_corpus_names
from .evaluation import get_references_path, get_estimates_path
from .evaluation import extract_tempo, extract_tempi_and_salience, \
    extract_tempo_from_beats, extract_c_var_from_beats, extract_tags, \
    is_mirex_style, is_single_bpm
from .evaluation import Metric, ACC1, ACC2, APE1, APE2, PE1, PE2, OE1, OE2, AOE1, AOE2,\
    PSCORE, ONE_CORRECT, BOTH_CORRECT
