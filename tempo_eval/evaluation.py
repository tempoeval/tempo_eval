"""
Core functions for evaluating tempo jams.

This module must not contain any formatting or printing logic.
"""
import logging
import math
import os
from bisect import bisect_left
from os import walk, listdir
from os.path import join, basename, dirname, exists, isdir
from statistics import stdev, mean
from typing import List, Callable, Any, Dict, Tuple, Iterable, Union, Set

import jams
import numpy as np
import pandas as pd
import statsmodels.stats.contingency_tables
from numba import njit, objmode
from scipy.stats import ttest_rel

import tempo_eval
from tempo_eval.parser.util import timestamps_to_bpm

logger = logging.getLogger('tempo_eval')

# type aliases used in functions (type hints)
# to make autodoc with sphinx a little easier
MirexTempo = Tuple[float, float, float]
PlainTempo = float
Tempo = Union[Any, MirexTempo, PlainTempo]
Tempi = Dict[str, Dict[str, Tempo]]
EvalResult = Any
EvalResults = Dict[str, Dict[str, Dict[str, EvalResult]]]
AverageResults = Dict[str, Dict[str, Tuple[List[float], List[float]]]]
Annotations = Dict[str, Dict[str, jams.Annotation]]
TagAnnotations = Dict[str, Annotations]


class Metric:
    """
    Metric.

    Structured collection of logic and injected functions for different metrics.

    :Example:

    >>> from tempo_eval import OE1, read_reference_annotations, read_estimate_annotations
    >>> gt_ref = read_reference_annotations('giantsteps_tempo', validate=False)
    >>> gt_est = read_estimate_annotations('giantsteps_tempo', validate=False)
    # evaluate estimates using the reference values and Metric OE1:
    >>> res = OE1.eval_annotations(gt_ref['tempo'], gt_est['tempo'])
    # show result of ref '1.0' and est 'davies2009/mirex_qm_tempotracker'
    # for file '3453642.LOFI.jams':
    >>> res['1.0']['davies2009/mirex_qm_tempotracker']['3453642.LOFI.jams']
    [-0.02271693594286862]
    """

    def __init__(self,
                 name: str,
                 formatted_name: str = None,
                 description: str = None,
                 eval_function: Callable[[Tempo, Tempo], EvalResult] = None,
                 extract_function: Callable[[jams.Annotation], Tempo] = None,
                 suitability_function: Callable[[Tempo], bool] = lambda x: True,
                 significant_difference_function:
                 Callable[[Dict[str, List[EvalResult]], Dict[str, List[EvalResult]], str, str],
                          List[float]] = None,
                 best_value: float = 1.,
                 signed: bool = False,
                 unit: Union[None, str] = '%') -> None:
        """
        Create a metric using the given functions.

        :param name: name
        :param description: HTML-formatted, high level description
        :param formatted_name: HTML-formatted name
        :param eval_function: function that compares two tempi,
            may also accept a tolerance parameter, e.g. :py:func:`~tempo_eval.equal1`.
        :param extract_function: function to extract tempo values
            from annotation, e.g. :py:func:`~tempo_eval.extract_tempo`.
        :param significant_difference_function: function to determine significant differences.
            E.g. :py:func:`~tempo_eval.evaluation.mcnemar`.
        :param suitability_function: function to determine whether
            a tempo value is suitable for this metric, e.g.
            :py:func:`~tempo_eval.is_single_bpm`.
        :param best_value: best value (to show, if no values are available)
        :param unit: unit, e.g., ``'%'``
        :param signed: is the metric signed (e.g. percentage error)
            or absolute (e.g. Accuracy1)?
        """
        super().__init__()
        self.name = name
        self.best_value = best_value
        self.signed = signed
        if formatted_name:
            self.formatted_name = formatted_name
        else:
            self.formatted_name = name
        if description:
            self.description = description
        else:
            self.description = self.formatted_name
        self.eval_function = eval_function
        if not eval_function:
            raise ValueError('eval_function must not be None')
        self.extract_function = extract_function
        if not extract_function:
            raise ValueError('extract_function must not be None')
        self.unit = unit
        self.suitability_function = suitability_function
        self.significant_difference_function = significant_difference_function

    def __repr__(self):
        return self.name

    def _create_eval_functions(self, tolerances: Iterable[float])\
            -> List[Callable[[Tempo, Tempo], EvalResult]]:
        """
        Creates a list of functions that each take two arguments, but use
        different tolerances.

        :param tolerances: iterable of desired tolerances
        :return: list of evaluation functions
        :rtype: list[function]
        """
        if tolerances is None:
            return [self.eval_function]
        else:
            def fixed_tolerance(t):
                return lambda a, b: self.eval_function(a, b, t)
            return [fixed_tolerance(t) for t in tolerances]

    def is_tempo_suitable(self, tempo) -> bool:
        """
        Indicates whether a given tempo is suitable for this metric.
        This is determined with the instance's suitability function.

        :param tempo: tempo value
        :return: ``True`` or ``False``
        :rtype: bool
        """
        return self.suitability_function(tempo)

    def are_tempi_suitable(self, tempi: Tempi) -> bool:
        """
        Is at least one tempo *meant* for this metric?

        :param tempi: collection of tempi as provided by \
            :py:func:`~tempo_eval.evaluation.Metric.extract_tempi`
        :type tempi: dict[str, object)
        :return: ``True``, if *any* tempi are suitable
        :rtype: bool
        """
        return any([self.is_tempo_suitable(t) for t in tempi.values()])

    def extract_tempi(self, annotation_set: Annotations) -> Tempi:
        """
        Extracts tempi from the given annotation set. This possibly
        extracts multiple values per track.
        Uses the ``extract_function`` provided at initialization.

        :param annotation_set: annotation set
        :return: dict of dicts with the version/annotationset name on
            the first level and item ids and their
            tempi on the second level
        :rtype: dict[str, dict[str, object]]
        """
        return {version: {item_id: self.extract_function(annotation) for item_id, annotation in annotations.items()}
                for version, annotations in annotation_set.items()}

    def eval_annotations(self, reference_annotation_set: Annotations,
                         estimates_annotation_set: Annotations,
                         tolerances: Iterable[float] = None) -> EvalResults:
        """
        Evaluates annotations.

        :param reference_annotation_set: reference annotations
        :type reference_annotation_set: dict[str, dict[str, jams.Annotation)
        :param estimates_annotation_set: estimates
        :type estimates_annotation_set: dict[str, dict[str, jams.Annotation)
        :param tolerances: array of tolerances (for metrics that need such a thing)
        :return: evaluation results (per track)
        :rtype: dict[str, dict[str, dict[str, object]]]
        """
        reference_tempi = self.extract_tempi(reference_annotation_set)
        estimated_tempi = self.extract_tempi(estimates_annotation_set)
        return self.eval_tempi(reference_tempi, estimated_tempi, tolerances)

    def eval_tempi(self, reference_tempi: Tempi,
                   estimated_tempi: Tempi,
                   tolerances: Iterable[float] = None) -> EvalResults:
        """
        Evaluates tempi for all provided tempi for each track.

        :param reference_tempi: tempi as provided by \
            :py:func:`~tempo_eval.evaluation.Metric.extract_tempi`
        :param estimated_tempi: tempi as provided by
            :py:func:`~tempo_eval.evaluation.Metric.extract_tempi`
        :param tolerances: array of tolerances to pass to the eval
            function as third parameter
        :return: evaluation results (per track) as nested dict with
            ground truth version, estimator name and item id as keys
        :rtype: dict[str, dict[str, dict[str, object]]]
        """
        eval_functions = self._create_eval_functions(tolerances)
        result = {}
        for groundtruth_version, specific_reference_tempi in reference_tempi.items():
            for item_id, reference_tempo in specific_reference_tempi.items():
                for estimator, specific_estimated_tempi in estimated_tempi.items():

                    if groundtruth_version not in result:
                        result[groundtruth_version] = {}
                    if estimator not in result[groundtruth_version]:
                        result[groundtruth_version][estimator] = {}

                    if item_id in specific_estimated_tempi:
                        estimated_tempo = specific_estimated_tempi[item_id]
                        if isinstance(estimated_tempo, float) and estimated_tempo <= 0.0:
                            logger.warning('Estimate {} BPM by \'{}\' for \'{}\' is <= 0.0 BPM. '
                                           'Reference tempo in \'{}\' '
                                           'is {} BPM. '
                                           'We will ignore this estimate and treat it as \'no value\'.'
                                           .format(estimated_tempo,
                                                   estimator, item_id,
                                                   groundtruth_version,
                                                   reference_tempo))
                            estimated_tempo = None
                        # TODO: What, if it's a tuple for PScore, not a float?
                    else:
                        logger.warning('Failed to find item \'{}\' in estimates by \'{}\'.'
                                       .format(item_id, estimator))
                        estimated_tempo = None

                    comparison_results = [eval_function(reference_tempo,
                                                        estimated_tempo)
                                          for eval_function in eval_functions]
                    if estimated_tempo is not None and np.isnan(np.sum(comparison_results)):
                        with objmode():
                            logger.warning('Metric {} returned nan: '
                                           'item={}, estimator={}, estimate={} BPM, '
                                           'reference_version={}, reference={} BPM'
                                           .format(self.name,
                                                   item_id, estimator, estimated_tempo,
                                                   groundtruth_version, reference_tempo))
                    result[groundtruth_version][estimator][item_id] = comparison_results
        return result

    def averages(self, eval_results: EvalResults,
                 item_id_filter: Callable[[str, str], bool] = lambda x, y: True,
                 undefined_value: float = None) -> AverageResults:
        """
        Calculate means and standard deviations for the given evaluation results.
        Possible INF and NaN values will be masked, i.e., ignored.
        This means that if an algorithms did not produce a valid estimate, averages
        and standard deviations will be computed without that value.

        :param undefined_value: value to use, if no data is available. Defaults to
            ``self.best_value``.
        :param eval_results: results as returned by
            :py:func:`~tempo_eval.evaluation.Metric.eval_tempi`
        :param item_id_filter: function taking reference name and item id as arguments to filter item ids
        :return: mean and standard deviation as nested dict with
            ground truth version and estimator name as keys
        :rtype: dict[str, dict[str, (list[float], list[float])]]
        """
        if undefined_value is None:
            undefined_value = self.best_value
        averages = {}
        for groundtruth_version, algorithms in eval_results.items():
            for estimator, results in algorithms.items():
                if groundtruth_version not in averages:
                    averages[groundtruth_version] = {}
                filtered_values = np.array([value
                                            for key, value in results.items()
                                            if item_id_filter(groundtruth_version, key)])
                if filtered_values.shape[0] == 0:
                    averages[groundtruth_version][estimator] = [undefined_value], [undefined_value]
                    continue

                # mask invalid, because APE can be INF and OE can be NaN
                masked_invalid_filtered_values = np.ma.masked_invalid(filtered_values)
                m = np.asarray(np.mean(masked_invalid_filtered_values, axis=0))
                s = np.asarray(np.std(masked_invalid_filtered_values, axis=0))
                averages[groundtruth_version][estimator] = m, s
        return averages


@njit
def equal1(reference_tempo: PlainTempo,
           estimated_tempo: Union[PlainTempo, None],
           tolerance: float = 0.04,
           factor: float = 1.0) -> bool:
    """
    Determines whether two tempi are considered *equal*, given an allowed tolerance
    and factor.

    When averaged, results correspond to *Accuracy 1*.

    See also :py:func:`~tempo_eval.equal2`.

    :param reference_tempo: references tempo
    :param estimated_tempo: estimated tempo
    :param tolerance: tolerance, default is 0.04, i.e., 4%
    :param factor: allowed deviation factor
    :return: ``True`` or ``False``
    :rtype: bool

    .. seealso:: Fabien Gouyon, Anssi P. Klapuri, Simon Dixon, Miguel Alonso,
        George Tzanetakis, Christian Uhle, and Pedro Cano. `An experimental
        comparison of audio tempo induction algorithms.
        <https://www.researchgate.net/profile/Fabien_Gouyon/publication/3457642_An_experimental_comparison_of_audio_tempo_induction_algorithms/links/0fcfd50d982025360f000000/An-experimental-comparison-of-audio-tempo-induction-algorithms.pdf>`_
        IEEE Transactions on Audio, Speech, and Language Processing,
        14(5):1832– 1844, 2006.
    """
    if estimated_tempo is None:
        return False
    if tolerance < 0 or tolerance > 1:
        raise ValueError('Tolerance must be in [0, 1]')
    return abs(reference_tempo*factor-estimated_tempo) <= (reference_tempo*factor * tolerance)


@njit
def equal2(reference_tempo: PlainTempo,
           estimated_tempo: Union[PlainTempo, None],
           tolerance: float = 0.04) -> bool:
    """
    Determines whether two tempi are considered *equal*, given an allowed tolerance
    and the factors 1, 2, 3, 1/2, and 1/3.

    When averaged, results correspond to *Accuracy 2*.

    See also :py:func:`~tempo_eval.equal1`.

    :param reference_tempo: reference tempo
    :param estimated_tempo: estimated tempo
    :param tolerance: tolerance, default is ``0.04``, i.e., 4%
    :return: ``True`` or ``False``
    :rtype: bool

    .. seealso:: Fabien Gouyon, Anssi P. Klapuri, Simon Dixon, Miguel Alonso,
        George Tzanetakis, Christian Uhle, and Pedro Cano. `An experimental
        comparison of audio tempo induction algorithms.
        <https://www.researchgate.net/profile/Fabien_Gouyon/publication/3457642_An_experimental_comparison_of_audio_tempo_induction_algorithms/links/0fcfd50d982025360f000000/An-experimental-comparison-of-audio-tempo-induction-algorithms.pdf>`_
        IEEE Transactions on Audio, Speech, and Language Processing,
        14(5):1832– 1844, 2006.
    """
    return equal1(reference_tempo, estimated_tempo, tolerance, 1.0) \
           or equal1(reference_tempo, estimated_tempo, tolerance, 2.0) \
           or equal1(reference_tempo, estimated_tempo, tolerance, 3.0) \
           or equal1(reference_tempo, estimated_tempo, tolerance, 1.0 / 2.0) \
           or equal1(reference_tempo, estimated_tempo, tolerance, 1.0 / 3.0)


@njit
def p_score(reference_tempo: MirexTempo,
            estimated_tempo: Union[MirexTempo, None],
            tolerance: float = 0.08) -> float:
    """
    P-Score is the weighted average of two tempi values.
    The weighting is based on a salience value (part of the ground truth).

    See also :py:func:`~tempo_eval.one_correct` and
    :py:func:`~tempo_eval.both_correct`.

    :param reference_tempo: (t1, t2, s1)
    :param estimated_tempo: (t1, t2, s1)
    :param tolerance: tolerance, e.g., ``0.08``
    :return: score :math:`\in [0,1]`
    :rtype: float

    .. seealso::
        * McKinney, M. F., Moelants, D., Davies, M. E., and Klapuri, A. P. (2007).
          `Evaluation of audio beat tracking and music tempo extraction algorithms.
          <http://www.cs.tut.fi/sgn/arg/klap/mckinney_jnmr07.pdf>`_
          Journal of New Music Research, 36(1):1–16.
        * `MIREX Audio Tempo Extraction 2006
          <https://www.music-ir.org/mirex/wiki/2006:Audio_Tempo_Extraction>`_
    """
    return _mirex(reference_tempo,
                  estimated_tempo,
                  tolerance=tolerance)[0]


@njit
def one_correct(reference_tempo: MirexTempo,
                estimated_tempo: Union[MirexTempo, None],
                tolerance: float = 0.08) -> bool:
    """
    Fraction of estimates with at least one correct tempo value.

    See also :py:func:`~tempo_eval.p_score` and
    :py:func:`~tempo_eval.both_correct`.

    :param reference_tempo: (t1, t2, s1)
    :param estimated_tempo: (t1, t2, s1)
    :param tolerance: tolerance, e.g., ``0.08``
    :return: fraction :math:`\in [0,1]`
    :rtype: float
    """
    return _mirex(reference_tempo,
                  estimated_tempo,
                  tolerance=tolerance)[1]


@njit
def both_correct(reference_tempo: MirexTempo,
                 estimated_tempo: Union[MirexTempo, None],
                 tolerance: float = 0.08) -> bool:
    """
    Fraction of estimates with two correct tempo values.

    See also :py:func:`~tempo_eval.p_score` and
    :py:func:`~tempo_eval.one_correct`.

    :param reference_tempo: (t1, t2, s1)
    :param estimated_tempo: (t1, t2, s1)
    :param tolerance: tolerance, e.g., ``0.08``
    :return: fraction :math:`\in [0,1]`
    :rtype: float
    """
    return _mirex(reference_tempo,
                  estimated_tempo,
                  tolerance=tolerance)[2]


@njit
def _mirex(reference_tempo: MirexTempo,
           estimated_tempo: Union[MirexTempo, None],
           tolerance=0.08) -> (float, bool, bool):

    if estimated_tempo is None:
        return 0., False, False
    else:
        # for now call our own implementation, to avoid stumbling over
        # 0 values. See https://github.com/craffel/mir_eval/issues/298
        return _mir_eval_detection(reference_tempo[0:2], reference_tempo[2],
                                   estimated_tempo[0:2], tol=tolerance)
        # but we actually want to call mir_eval.tempo.detection
        # return mir_eval.tempo.detection(np.array([reference_tempo[0], reference_tempo[1]]), reference_tempo[2],
        #                                 np.array([estimated_tempo[0], estimated_tempo[1]]), tol=tolerance)


@njit
def _mir_eval_detection(reference_tempi: MirexTempo,
                        reference_weight: float,
                        estimated_tempi: MirexTempo,
                        tol: float = 0.08) -> (float, bool, bool):
    """
    Function copied from mir_eval.tempo.detection minus the validation.

    :param reference_tempi: np.ndarray, shape=(2,)
        Two non-negative reference tempi
    :param reference_weight: float > 0
        The relative strength of ``reference_tempi[0]`` vs
        ``reference_tempi[1]``.
    :param estimated_tempi: np.ndarray, shape=(2,)
        Two non-negative estimated tempi.
    :param tol: float in [0, 1]:
        The maximum allowable deviation from a reference tempo to
        count as a hit.
        ``|est_t - ref_t| <= tol * ref_t``
        (Default value = 0.08)
    :return: p_score, one_correct, both_correct
    :rtype: (float, bool, bool)
    """
    if tol < 0 or tol > 1:
        raise ValueError('invalid tolerance {}: must lie in the range '
                         '[0, 1]')

    hits = [False, False]

    # avoid numpy, as it is slow for such small arrays, roughly factor 4
    for i, ref_t in enumerate(reference_tempi):
        if ref_t > 0:
            # Compute the relative error for this reference tempo
            relative_error = min(
                abs(ref_t - estimated_tempi[0]),
                abs(ref_t - estimated_tempi[1])
            ) / float(ref_t)
            # Count the hits
            hits[i] = relative_error <= tol

    score = reference_weight * hits[0] + (1.0-reference_weight) * hits[1]
    one = hits[0] or hits[1]
    both = hits[0] == hits[1] and hits[0]

    return score, one, both


@njit
def pe1(reference_tempo: PlainTempo,
        estimated_tempo: Union[PlainTempo, None],
        factor: float = 1.0) -> float:
    """
    Percentage error for two tempi values allowing for a given factor.
    If a reference tempo of 0 is given, this function returns NaN.
    If an estimate of None is given, this function returns NaN.

    See also :py:func:`~tempo_eval.pe2`.

    :param reference_tempo: a reference tempo, should not be 0
    :param estimated_tempo: an estimated tempo
    :param factor: multiplication factor
    :return: the percentage error
    :rtype: float
    """
    if estimated_tempo is None:
        return np.nan
    if reference_tempo == 0.:
        return np.nan
    else:
        return (estimated_tempo*factor-reference_tempo)/reference_tempo


@njit
def pe2(reference_tempo: PlainTempo,
        estimated_tempo: Union[PlainTempo, None]) -> float:
    """
    Percentage error for two tempi values allowing the factors
    1, 2, 3, 1/2, and 1/3 with the smallest absolute value.
    If an estimate of None is given, this function returns NaN.

    See also :py:func:`~tempo_eval.pe1`.

    :param reference_tempo: a references tempo, must not be 0
    :param estimated_tempo: an estimated tempo
    :return: the percentage error for the factors 1, 2, 3, 1/2, and 1/3 with the smallest absolute value
    :rtype: float
    """

    if estimated_tempo is None:
        return np.nan

    pe1_1 = pe1(reference_tempo, estimated_tempo)
    pe1_2 = pe1(reference_tempo, estimated_tempo, factor=2.)
    pe1_12 = pe1(reference_tempo, estimated_tempo, factor=0.5)
    pe1_3 = pe1(reference_tempo, estimated_tempo, factor=3.)
    pe1_13 = pe1(reference_tempo, estimated_tempo, factor=1. / 3.)

    pe1s = {abs(pe1_1): pe1_1,
            abs(pe1_2): pe1_2,
            abs(pe1_12): pe1_12,
            abs(pe1_3): pe1_3,
            abs(pe1_13): pe1_13}

    return pe1s[min(pe1s.keys())]


@njit
def ape1(reference_tempo: PlainTempo,
         estimated_tempo: Union[PlainTempo, None],
         factor: float = 1.0) -> float:
    """
    Absolute percentage error for two tempi values allowing for a given factor.
    If a reference tempo of 0 is given, this function returns NaN.
    If an estimate of None is given, this function returns NaN.

    When averaged, results correspond to *MAPE1*.

    See also :py:func:`~tempo_eval.ape2`.

    :param reference_tempo: a reference tempo, should not be 0
    :param estimated_tempo: an estimated tempo
    :param factor: multiplication factor
    :return: the absolute percentage error
    :rtype: float
    """

    if estimated_tempo is None:
        return np.nan

    return abs(pe1(reference_tempo=reference_tempo,
                   estimated_tempo=estimated_tempo,
                   factor=factor))


@njit
def ape2(reference_tempo: PlainTempo,
         estimated_tempo: Union[PlainTempo, None]) -> float:
    """
    Minimum of the absolute percentage error for two tempi values allowing the factors
    1, 2, 3, 1/2, and 1/3.
    If an estimate of None is given, this function returns NaN.

    When averaged, results correspond to *MAPE2*.

    See also :py:func:`~tempo_eval.ape1`.

    :param reference_tempo: a references tempo, must not be 0
    :param estimated_tempo: an estimated tempo
    :return: the minimal absolute percentage error for the factors 1, 2, 3, 1/2, and 1/3.
    :rtype: float
    """

    if estimated_tempo is None:
        return np.nan

    return min(
        ape1(reference_tempo, estimated_tempo),
        ape1(reference_tempo, estimated_tempo, factor=2.),
        ape1(reference_tempo, estimated_tempo, factor=0.5),
        ape1(reference_tempo, estimated_tempo, factor=3.),
        ape1(reference_tempo, estimated_tempo, factor=1. / 3.)
    )


@njit
def oe1(reference_tempo: PlainTempo,
        estimated_tempo: Union[PlainTempo, None],
        factor: float = 1.0) -> float:
    """
    Octave error for two tempi values allowing for a given factor.
    If a reference or estimated tempo of 0 is given, this function returns NaN.
    If an estimate of None is given, this function returns NaN.

    See also :py:func:`~tempo_eval.oe2`.

    :param reference_tempo: a reference tempo, should not be 0
    :param estimated_tempo: an estimated tempo
    :param factor: multiplication factor
    :return: the octave error
    :rtype: float
    """
    if estimated_tempo is None:
        return np.nan
    elif reference_tempo == 0.:
        return np.nan
    elif estimated_tempo == 0.:
        return np.nan
    else:
        return np.log2((estimated_tempo * factor) / reference_tempo)


@njit
def oe2(reference_tempo: PlainTempo,
        estimated_tempo: Union[PlainTempo, None]) -> float:
    """
    Octave error for two tempi values allowing the factors
    1, 2, 3, 1/2, and 1/3 with the smallest absolute value.
    If an estimate of None is given, this function returns NaN.

    See also :py:func:`~tempo_eval.oe1`.

    :param reference_tempo: a references tempo, must not be 0
    :param estimated_tempo: an estimated tempo
    :return: the percentage error for the factors 1, 2, 3, 1/2, and 1/3 with the smallest absolute value
    :rtype: float
    """

    if estimated_tempo is None:
        return np.nan

    oe1_1 = oe1(reference_tempo, estimated_tempo)
    oe1_2 = oe1(reference_tempo, estimated_tempo, factor=2.)
    oe1_12 = oe1(reference_tempo, estimated_tempo, factor=0.5)
    oe1_3 = oe1(reference_tempo, estimated_tempo, factor=3.)
    oe1_13 = oe1(reference_tempo, estimated_tempo, factor=1. / 3.)

    oe1s = {abs(oe1_1): oe1_1,
            abs(oe1_2): oe1_2,
            abs(oe1_12): oe1_12,
            abs(oe1_3): oe1_3,
            abs(oe1_13): oe1_13}

    return oe1s[min(oe1s.keys())]


@njit
def aoe1(reference_tempo: PlainTempo,
         estimated_tempo: Union[PlainTempo, None],
         factor: float = 1.0) -> float:
    """
    Absolute octave error for two tempi values allowing for a given factor.
    If a reference or estimated tempo of 0 is given, this function returns NaN.
    If an estimate of None is given, this function returns NaN.

    When averaged, results correspond to *MAOE1*.

    See also :py:func:`~tempo_eval.aoe2`.

    :param reference_tempo: a reference tempo, should not be 0
    :param estimated_tempo: an estimated tempo, should not be 0
    :param factor: multiplication factor
    :return: the absolute octave error
    :rtype: float
    """
    if estimated_tempo is None:
        return np.nan
    return abs(oe1(reference_tempo=reference_tempo,
                   estimated_tempo=estimated_tempo,
                   factor=factor))


@njit
def aoe2(reference_tempo: PlainTempo,
         estimated_tempo: Union[PlainTempo, None]) -> float:
    """
    Minimum of the absolute octave error for two tempi values allowing the factors
    1, 2, 3, 1/2, and 1/3.
    If an estimate of None is given, this function returns NaN.

    When averaged, results correspond to *MAOE2*.

    See also :py:func:`~tempo_eval.aoe1`.

    :param reference_tempo: a references tempo, must not be 0
    :param estimated_tempo: an estimated tempo, must not be 0
    :return: the minimal absolute percentage error for the factors 1, 2, 3, 1/2, and 1/3.
    :rtype: float
    """
    if estimated_tempo is None:
        return np.nan

    return min(
        aoe1(reference_tempo, estimated_tempo),
        aoe1(reference_tempo, estimated_tempo, factor=2.),
        aoe1(reference_tempo, estimated_tempo, factor=0.5),
        aoe1(reference_tempo, estimated_tempo, factor=3.),
        aoe1(reference_tempo, estimated_tempo, factor=1. / 3.)
    )


def is_mirex_style(t: Tempo) -> bool:
    """
    Does the given tempo have MIREX style, i.e., ``(t1, t2, s1)`` with
    t1 and t2 not equal to 0 and s1 neither 0 or 1.
    Returns ``False`` for values equal to or less than 0 and greater than 5000.
    The upper bound is chosen somewhat arbitrarily.
    Note that a value like ``(0, 100, 0)`` is not good enough.

    :param t: tempo list-like object
    :return: ``True`` or ``False``
    :rtype: bool
    """
    try:
        return len(t) == 3 \
               and 0. < t[0] <= 5000. \
               and 0. < t[1] <= 5000. \
               and t[0] < t[1] \
               and 0. < t[2] < 1.
    except:
        return False


def is_single_bpm(t: Tempo) -> bool:
    """
    Is the given tempo a plausible BPM value?
    Returns ``False`` for values less than 0 and greater than 5000.
    The upper bound is chosen somewhat arbitrarily.

    :param t: tempo
    :return: ``True`` or ``False``
    :rtype: bool
    """
    try:
        return 0. <= t <= 5000.
    except:
        return False


def get_data(file_name: str) -> str:
    """
    Get real path for a packaged data file.

    :param file_name: file name
    :return: fill path
    """
    packagedir = tempo_eval.__path__[0]
    fullname = join(dirname(packagedir), file_name)
    return fullname


def get_references_path(dataset: str = None,
                        folder_name: str = None,
                        file_name: str = None) -> str:
    """
    Get real path for a reference data set.

    :param dataset: dataset name
    :param folder_name: folder name
    :param file_name: file name
    :return: path
    """
    rel_path = join('annotations', 'references', dataset, folder_name)
    if file_name is not None:
        rel_path = join(rel_path, file_name)
        return get_data(rel_path)
    else:
        return get_data(rel_path) + os.sep


def get_estimates_path(dataset: str = None,
                       folder_name: str = None,
                       file_name: str = None) -> str:
    """
    Get real path for an estimates data set.

    :param dataset: dataset name
    :param folder_name: folder name
    :param file_name: file name
    :return: path
    """
    rel_path = join('annotations', 'estimates', dataset)
    if folder_name is not None:
        rel_path = join(rel_path, folder_name)
    if file_name is not None:
        rel_path = join(rel_path, file_name)
        return get_data(rel_path)
    else:
        return get_data(rel_path) + os.sep


def list_reference_corpus_names() -> List[str]:
    """
    List of corpus names for (built-in) references datasets.
    Instead of the actual jam annotation_metadata corpus name,
    we use the names of the base directories of the references
    repository, as this allows us listing them without parsing
    the whole tree.

    Additionally, using directory names lets us better control
    the spelling of corpus names, as we don't have to rely on
    what's in many many jam files.

    :return: sorted list of corpus names for reference datasets
    :rtype: list[str]

    :Example:

    >>> from tempo_eval import list_reference_corpus_names
    >>> list_reference_corpus_names()
    ['acm_mirum', 'ballroom', 'beatles', 'fma_medium', 'fma_small',
    'giantsteps_mtg_key', 'giantsteps_tempo', 'gtzan', 'hainsworth',
    'ismir2004songs', 'klapuri', 'lmd_tempo', 'rwc_mdb_c', 'rwc_mdb_g',
    'rwc_mdb_j', 'rwc_mdb_p', 'rwc_mdb_r', 'smc_mirex', 'wjd']

    .. seealso:: To actually read annotations, use
        :py:func:`~tempo_eval.read_reference_annotations`.
    """
    path = join(dirname(tempo_eval.__path__[0]), 'annotations', 'references')
    return sorted([e for e in listdir(path) if not e.startswith('.') and isdir(join(path, e))])


def list_estimate_corpus_names() -> List[str]:
    """
    List of corpus names for (built-in) estimates.
    Instead of the actual jam annotation_metadata corpus name,
    we use the names of the base directories of the estimates
    repository, as this allows us listing them without parsing
    the whole tree.

    Additionally, using directory names lets us better control
    the spelling of corpus names, as we don't have to rely on
    what's in many many jam files.

    :return: sorted list of corpus names for estimate datasets
    :rtype: list[str]

    :Example:

    >>> from tempo_eval import list_estimate_corpus_names
    >>> list_estimate_corpus_names()
    ['acm_mirum', 'ballroom', 'beatles', 'fma_medium', 'fma_small',
    'giantsteps_mtg_key', 'giantsteps_tempo', 'gtzan', 'hainsworth',
    'ismir2004songs', 'lmd_tempo', 'queen', 'rwc_mdb_c', 'rwc_mdb_g',
    'rwc_mdb_j', 'rwc_mdb_p', 'rwc_mdb_r', 'smc_mirex', 'wjd']

    .. seealso:: To actually read estimates, use
        :py:func:`~tempo_eval.read_estimate_annotations`.
    """
    path = join(dirname(tempo_eval.__path__[0]), 'annotations', 'estimates')
    return sorted([e for e in listdir(path) if not e.startswith('.') and isdir(join(path, e))])


def read_annotations(path: str,
                     derive_version: Callable[[str, jams.Annotation], str] = None,
                     namespace: Union[str, Iterable[str]] = 'tempo',
                     derive_item_id: Callable[[str, jams.JAMS], str] = lambda file, jam: basename(file),
                     validate: bool = True,
                     split_by_corpus: bool = False) -> Union[Dict[str, Annotations], Dict[str, Dict[str, Annotations]]]:
    """
    Recursively read all jam files from the given directory.

    :param path: base directory
    :param derive_version: function that derives a version a given annotation
    :param namespace: one or more jam annotation namespace(s), e.g., ``tempo`` for tempo annotations
    :param derive_item_id: function that returns an id given a file name and jams
    :param validate: validate jam while reading (validation impacts performance negatively)
    :param split_by_corpus: wrap results in a dict with corpus names (extracted from jams) as keys
    :return: jam annotation objects, organized as nested dicts with version and item ids as keys,
        if ``split_by_corpus`` the outermost dict uses corpus names as keys
    :raises FileNotFoundError: if ``path`` does not exist or is not a directory
    """
    logger.debug('Reading annotations from \'{}\' ...'.format(path))

    if not exists(path):
        raise FileNotFoundError('Annotations path does not exist: {}'.format(path))
    if not isdir(path):
        raise FileNotFoundError('Annotations path is not a directory: {}'.format(path))

    annotations = {}
    for (dirpath, _, file_names) in walk(path):
        for file_name in [f for f in file_names if f.endswith('.jams')]:
            jam_file_name = join(dirpath, file_name)
            try:
                jam = jams.load(jam_file_name, validate=validate)
                item_id = derive_item_id(jam_file_name, jam)
                namespace_iterable = [namespace] if isinstance(namespace, str) else namespace

                for ns in namespace_iterable:
                    for annotation in jam.annotations[ns]:
                        # normalize corpus name slightly..
                        corpus = annotation.annotation_metadata.corpus.lower().replace(' ', '_')

                        if split_by_corpus and corpus not in annotations:
                            annotations[corpus] = {}

                        base = annotations[corpus] if split_by_corpus else annotations

                        if ns not in base:
                            base[ns] = {}
                        if derive_version is None:
                            version = _get_version(annotation, split_by_corpus)
                        else:
                            version = derive_version(jam_file_name, annotation)

                        if version not in base[ns]:
                            base[ns][version] = {}
                        if item_id in base[ns][version]:
                            logger.warning('Found multiple \'{}\'-annotations with the same version ({}) for '
                                           'item \'{}\'. Ignoring all but the first one.'
                                           .format(ns, version, item_id))
                        else:
                            base[ns][version][item_id] = annotation
            except Exception as e:
                logger.error('Error while parsing JAMS file {}: {}'
                             .format(jam_file_name, e))
                raise e

    return annotations


def read_reference_annotations(corpus_name: str,
                               namespace: Union[str, Iterable[str]] = 'tempo',
                               validate: bool = True) -> Dict[str, Annotations]:
    """
    Read annotations for reference datasets.

    :param corpus_name: corpus name (corresponds to folder name).
        See :py:func:`~tempo_eval.list_reference_corpus_names` to get a list of valid names.
    :param namespace: one or more jam annotation namespace(s), e.g., ``tempo`` for tempo annotations
        (see `jams namespaces <https://jams.readthedocs.io/en/stable/namespace.html#namespace>`_)
    :param validate: validate jam while reading (validation impacts performance negatively)
    :return: jam annotation objects, organized as nested dicts with version and item ids as keys
    :rtype: dict[str, dict[str, jams.Annotation]]

    .. note:: By default, `validate` is `True` in order to stay safe.
        But since it affects performance quite a bit, you might want to turn
        validation off when using this function to keep your sanity.

    :Example:

    >>> from tempo_eval import read_reference_annotations, extract_tempo
    >>> smc_ground_truth = read_reference_annotations('smc_mirex', validate=False)  # turn validation off for speed
    >>> smc_1_0_205 = smc_ground_truth['tempo']['1.0']['SMC_205.jams']  # choose reference version '1.0'
    >>> tempo = extract_tempo(smc_1_0_205)
    >>> print(tempo)
    78.74015748031492

    .. seealso:: To read estimates, use :py:func:`~tempo_eval.read_estimate_annotations`.
    """
    path = get_references_path(corpus_name, 'jams')

    def item_id(file_name, _):
        return file_name.replace(path, '')

    return read_annotations(path,
                            namespace=namespace,
                            derive_item_id=item_id,
                            validate=validate)


def read_estimate_annotations(corpus_name: str,
                              namespace: Union[str, Iterable[str]] = 'tempo',
                              validate: bool = True) -> Dict[str, Annotations]:
    """
    Read annotations for estimates.

    :param corpus_name: corpus name (corresponds to folder name).
        See :py:func:`~tempo_eval.list_estimate_corpus_names` to get a list of valid names.
    :param namespace: one or more jam annotation namespace(s), e.g., ``tempo`` for tempo annotations
        (see `jams namespaces <https://jams.readthedocs.io/en/stable/namespace.html#namespace>`_)
    :param validate: validate jam while reading (validation impacts performance negatively)
    :return: jam annotation objects, organized as nested dicts with version and item ids as keys
    :rtype: dict[str, dict[str, jams.Annotation]]

    .. note:: By default, `validate` is `True` in order to stay safe.
        But since it affects performance quite a bit, you might want to turn
        validation off when using this function to keep your sanity.

    :Example:

    >>> from tempo_eval import read_estimate_annotations, extract_tempo
    >>> smc_estimates = read_estimate_annotations('smc_mirex', validate=False)  # turn validation off for speed
    >>> smc_schreiber2014_205 = smc_estimates['tempo']['schreiber2014/default']['SMC_205.jams']
    >>> tempo = extract_tempo(smc_schreiber2014_205)
    >>> print(tempo)
    79.7638

    .. seealso:: To read *reference* values, use :py:func:`~tempo_eval.read_reference_annotations`.
    """
    path = get_estimates_path(corpus_name)

    def item_id(file_name, _):
        return os.sep.join(file_name.replace(path, '').split(os.sep)[2:])

    def corpus_version(file_name, _):
        return os.sep.join(file_name.replace(path, '').split(os.sep)[0:2])

    return read_annotations(path,
                            namespace=namespace,
                            derive_version=corpus_version,
                            derive_item_id=item_id,
                            validate=validate)


def read_reference_tags(corpus_name: str,
                        validate: bool = True) -> TagAnnotations:
    """
    Read reference tags from the namespaces 'tag_open', 'tag_gtzan', and 'tag_fma_genre'.

    :param corpus_name: corpus
        See :py:func:`~tempo_eval.list_reference_corpus_names` to get a list of valid names.
    :param validate: validate jam while reading (validation impacts performance negatively)
    :return: jam annotation objects, organized as nested dicts with namespace, version and item ids as keys
    :rtype: dict[str, dict[str, dict[str, jams.Annotation]]]

    .. note:: By default, `validate` is `True` in order to stay safe.
        But since it affects performance quite a bit, you might want to turn
        validation off when using this function to keep your sanity.

    :Example:

    >>> from tempo_eval import read_reference_tags, extract_tags
    >>> gtzan = read_reference_tags('gtzan', validate=False)
    >>> gtzan_hh_691694 = gtzan['tag_open']['GTZAN-Rhythm_v2_ismir2015_lbd_2015-10-28']['hiphop.00086.jams']
    >>> tags = extract_tags(gtzan_hh_691694)
    >>> print(tags)
    {'4/4', 'no_ternary', 'no_swing'}

    .. seealso:: To extract tag values, use :py:func:`~tempo_eval.extract_tags`.
    """
    tag_references = read_reference_annotations(corpus_name,
                                                namespace=['tag_open', 'tag_gtzan', 'tag_fma_genre'],
                                                validate=validate)
    return tag_references


def item_ids_for_differing_annotations(eval_results: EvalResults) -> Dict[str, Dict[str, List[str]]]:
    """
    Find item ids for differing annotations according the results of an evaluation
    that returns boolean values, like Accuracy1 or Accuracy2.

    :param eval_results: results as returned by
        :py:func:`~tempo_eval.evaluation.Metric.eval_tempi`
    :return: item ids of differing annotations as a nested dict with
        references and estimator names as keys.
    """
    result = {}
    for reference_version in eval_results.keys():
        result[reference_version] = {}
        for estimator in eval_results[reference_version].keys():
            for item_id in eval_results[reference_version][estimator].keys():
                if estimator not in result[reference_version]:
                    count = len(eval_results[reference_version][estimator][item_id])
                    result[reference_version][estimator] = [[] for _ in range(count)]
                for i, v in enumerate(eval_results[reference_version][estimator][item_id]):
                    if not v:
                        result[reference_version][estimator][i].append(item_id)
    return result


def significant_difference(metric: Metric, eval_results: EvalResults,
                           item_id_filter: Callable[[str], bool] = lambda x: True)\
        -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Calculate p-values of McNemar's test. Only applicable for binary
    metrics like Accuracy1 or Accuracy2, but not for MAPE.

    .. seealso:: Fabien Gouyon, Anssi P. Klapuri, Simon Dixon, Miguel Alonso,
        George Tzanetakis, Christian Uhle, and Pedro Cano. `An experimental
        comparison of audio tempo induction algorithms.
        <https://www.researchgate.net/profile/Fabien_Gouyon/publication/3457642_An_experimental_comparison_of_audio_tempo_induction_algorithms/links/0fcfd50d982025360f000000/An-experimental-comparison-of-audio-tempo-induction-algorithms.pdf>`_
        IEEE Transactions on Audio, Speech, and Language Processing,
        14(5):1832– 1844, 2006.

    :param metric: metric
    :param eval_results: results as returned by
        :py:func:`~tempo_eval.evaluation.Metric.eval_tempi`
    :param item_id_filter: boolean function that filters item ids
    :return: p-values in nested dicts with the ground truth,
        estimator 1 and estimator 2 as keys
    """
    pvalues = {}
    for groundtruth_key, algorithms in eval_results.items():
        pvalues[groundtruth_key] = {}
        # for each groundtruth
        for estimator1, results1 in algorithms.items():
            pvalues[groundtruth_key][estimator1] = {}
            filtered_results1 = _key_filter(results1, item_id_filter)
            for estimator2, results2 in algorithms.items():
                if estimator1 == estimator2 and filtered_results1:
                    # diagonal is always 1
                    ones = [1.0 for _ in range(len(list(filtered_results1.values())[0]))]
                    pvalues[groundtruth_key][estimator1][estimator2] = ones
                elif estimator2 in pvalues[groundtruth_key]:
                    # exploit symmetry
                    pvalues[groundtruth_key][estimator1][estimator2] = pvalues[groundtruth_key][estimator2][estimator1]
                else:
                    # actually do some computation
                    filtered_results2 = _key_filter(results2, item_id_filter)
                    pvalues[groundtruth_key][estimator1][estimator2]\
                        = metric.significant_difference_function(filtered_results1,
                                                                 filtered_results2,
                                                                 estimator1_name=estimator1,
                                                                 estimator2_name=estimator2)
    return pvalues


def _key_filter(dictionary: Dict[Any, Any], predicate: Callable[[Any], bool]) -> Dict[Any, Any]:
    """
    Utility function to filter a dict based on its keys.

    :param dictionary: dict
    :param predicate: predicate
    :return: filtered dict
    :rtype: dict
    """
    return {key: value for key, value in dictionary.items() if predicate(key)}


def mcnemar(estimator1_results: Dict[str, List[EvalResult]],
            estimator2_results: Dict[str, List[EvalResult]],
            estimator1_name: str = 'unknown_estimator1',
            estimator2_name: str = 'unknown_estimator2') -> List[float]:
    """
    Calculate McNemar's p-value for the results of two algorithms.

    .. seealso:: Fabien Gouyon, Anssi P. Klapuri, Simon Dixon, Miguel Alonso,
        George Tzanetakis, Christian Uhle, and Pedro Cano. `An experimental
        comparison of audio tempo induction algorithms.
        <https://www.researchgate.net/profile/Fabien_Gouyon/publication/3457642_An_experimental_comparison_of_audio_tempo_induction_algorithms/links/0fcfd50d982025360f000000/An-experimental-comparison-of-audio-tempo-induction-algorithms.pdf>`_
        IEEE Transactions on Audio, Speech, and Language Processing,
        14(5):1832– 1844, 2006.

    .. seealso:: `How to Calculate McNemar’s Test to Compare Two Machine Learning Classifiers
        <https://machinelearningmastery.com/mcnemars-test-for-machine-learning/>`_.

    :param estimator1_results: estimator 1 eval results
    :param estimator2_results: estimator 2 eval results
    :param estimator1_name: name for estimator 1
    :param estimator2_name: name for estimator 2
    :return: p values
    """
    contingency_tables = []
    for relative_jam_file_name in estimator1_results.keys():
        alg1_correct = estimator1_results[relative_jam_file_name]
        if relative_jam_file_name not in estimator2_results:
            logger.warning('Item only occurs in set \'{}\', but not in \'{}\': {}'
                           .format(estimator1_name, estimator2_name,
                                   relative_jam_file_name))
            continue
        alg2_correct = estimator2_results[relative_jam_file_name]
        for i in range(len(alg1_correct)):
            if len(contingency_tables) <= i:
                t = [[0, 0], [0, 0]]
                contingency_tables.append(t)
            contingency_tables[i][int(alg1_correct[i])][int(alg2_correct[i])] += 1
    # use binomial or chi_square distribution (threshold=25),
    # see e.g. https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
    result = [
        statsmodels.stats.contingency_tables.mcnemar(table,
                                                     exact=np.min(table) <= 25,
                                                     correction=np.min(table) > 25).pvalue
        for table in contingency_tables]

    return result


def ttest(estimator1_results: Dict[str, List[EvalResult]],
          estimator2_results: Dict[str, List[EvalResult]],
          estimator1_name: str = 'unknown_estimator1',
          estimator2_name: str = 'unknown_estimator2') -> List[float]:
    """
    Calculate t-test p-value for the results of two algorithms.

    :param estimator1_results: estimator 1 eval results
    :param estimator2_results: estimator 2 eval results
    :param estimator1_name: name for estimator 1
    :param estimator2_name: name for estimator 2
    :return: p values
    """
    a = []
    b = []
    for relative_jam_file_name in estimator1_results.keys():
        if relative_jam_file_name not in estimator2_results:
            logger.warning('Item only occurs in set \'{}\', but not in \'{}\': {}'
                           .format(estimator1_name, estimator2_name,
                                   relative_jam_file_name))
            continue
        a.append(estimator1_results[relative_jam_file_name])
        b.append(estimator2_results[relative_jam_file_name])

    np_a = np.vstack(a)
    np_b = np.vstack(b)
    ps = []
    for i in range(np_a.shape[1]):
        a1 = np_a[:, i]
        b1 = np_b[:, i]
        valid_indices = np.logical_and(np.isfinite(a1), np.isfinite(b1))
        a_valid = a1[valid_indices]
        b_valid = b1[valid_indices]
        if np.array_equal(a_valid, b_valid):
            p = 1.
        else:
            # ensure that we compare only valid values
            _, p = ttest_rel(a_valid, b_valid)
        ps.append(p)
    return ps


def _get_version(annotation: jams.Annotation, include_corpus: bool = False) -> str:
    """
    Get the annotation's version.
    If the version is missing, an id is artificially generated by
    creating a hash from the annotation metadata object.

    :param include_corpus: include corpus name
    :param annotation: jams annotation instance
    :return: a corpus id identifying corpus and version
    :rtype: str
    """
    v = annotation.annotation_metadata.version

    if not v:
        logger.warning('For a reference dataset, a version should be present to uniquely identify it. {}'
                       .format(annotation.annotation_metadata))
        # artificial corpus name, based on the hash value of the json dump
        v = 'unknown_version({})'.format(hash(annotation.annotation_metadata.dumps()))

    if include_corpus and annotation.annotation_metadata.corpus:
        v = annotation.annotation_metadata.corpus + '/' + v

    return v


def extract_tags(tag_annotations: jams.Annotation) -> Set[str]:
    """
    Extract tags from annotations object as set.

    :param tag_annotations: annotations
    :return: set of tags
    :rtype: set[str]

    :Example:

    >>> from tempo_eval import read_reference_annotations, extract_tags
    # choose namespace 'tag_gtzan'!
    >>> gtzan = read_reference_annotations('gtzan', namespace='tag_gtzan', validate=False)
    >>> gtzan_hh_691694 = gtzan['tag_gtzan']['1.0']['hiphop.00086.jams']
    >>> tags = extract_tags(gtzan_hh_691694)
    >>> print(tags)
    {'hip-hop'}

    """
    return set([observation.value for observation in (tag_annotations['data'])])


def extract_tempo(tempo_annotations: jams.Annotation) -> PlainTempo:
    """
    Extract the most salient (greatest confidence) tempo value from the annotation.

    :param tempo_annotations: annotations
    :return: a tempo value (typically in BPM)
    :rtype: float

    :Example:

    >>> from tempo_eval import read_reference_annotations, extract_tempo
    >>> smc_ground_truth = read_reference_annotations('smc_mirex', validate=False)  # turn validation off for speed
    >>> smc_1_0_205 = smc_ground_truth['tempo']['1.0']['SMC_205.jams']  # choose reference version '1.0'
    >>> tempo = extract_tempo(smc_1_0_205)  # extract single tempo from JAMS annotation
    >>> print(tempo)
    78.74015748031492
    """
    observations = tempo_annotations['data']
    observation_length = len(observations)

    if observation_length == 1:
        reference_tempo = observations[0].value
    elif observation_length == 2:
        o0 = observations[0]
        o1 = observations[1]
        if o0.confidence >= o1.confidence:
            reference_tempo = o0.value
        else:
            reference_tempo = o1.value
    elif observation_length == 0:
        raise ValueError('Encountered tempo annotation with no observation in {}'.format(tempo_annotations))
    else:
        raise ValueError('Don\'t know what to do with more than two tempo annotations. Jams: {}'
                         .format(tempo_annotations))

    return reference_tempo


def extract_tempi_and_salience(tempo_annotations: jams.Annotation) -> MirexTempo:
    """
    Extract MIREX-style values from the annotations. I.e. ``(t1, t2, s1)``
    --- tempo 1, tempo 2 and the salience value for tempo1.
    Tempo values are ordered: ``t1 < t2``.

    :param tempo_annotations: annotations
    :return: ``t1, t2, s1``
    :rtype: (float, float, float)

    :Example:

    >>> from tempo_eval import read_reference_annotations, extract_tempi_and_salience
    >>> gt = read_reference_annotations('smc_mirex', validate=False)  # turn validation off for speed
    >>> gt_691694 = gt['tempo']['2.0']['691694.LOFI.jams']  # choose reference version '2.0'
    >>> tempo = extract_tempi_and_salience(gt_691694)  # extract single tempo from JAMS annotation
    >>> print(tempo)
    (73.0, 144.0, 0.04228329809725159)

    """
    observations = tempo_annotations['data']
    tempo2 = 0.0
    salience1 = 1.0
    observation_length = len(observations)
    if observation_length == 1:
        tempo1 = observations[0].value
    elif observation_length == 2:
        tempo1 = observations[0].value
        tempo2 = observations[1].value
        salience1 = observations[0].confidence
    elif observation_length == 0:
        raise ValueError('Encountered tempo annotation with no observation in {}'
                         .format(tempo_annotations))
    else:
        raise ValueError('Don\'t know what to do with more than two tempo annotations. Jams: {}'
                         .format(tempo_annotations))
    # MIREX ordering rule
    if tempo1 < tempo2:
        return tempo1, tempo2, salience1
    else:
        return tempo2, tempo1, 1.-salience1


def extract_tempo_from_beats(beat_annotations: jams.Annotation) -> Tuple[PlainTempo, PlainTempo, float, float]:
    """
    Extract tempo values from beat annotations using :py:func:`~tempo_eval.evaluation.timestamps_to_bpm`.

    :param beat_annotations: annotations
    :return: a BPM value
    """
    observations = beat_annotations['data']
    timestamps = [o.time for o in observations]
    values = [int(round(o.value)) for o in observations if not math.isnan(o.value)]
    values_length = len(values)
    if values_length > 20:
        meter = max(1, max(values[:20]))
    elif values_length > 0:
        meter = max(1, max(values))
    else:
        meter = 1
    return timestamps_to_bpm(timestamps, meter=meter)


def extract_c_var_from_beats(annotation_set: Dict[str, Dict[str, jams.Annotation]]) -> Dict[str, Dict[str, float]]:
    """
    Extract coefficient of variations for the beats in the annotation set.

    :param annotation_set: dict of annotations with annotation name as key
    :return: nested dicts with set and item names as key and normalized
        tempo standard deviations as value
    """
    result = {}
    for version, annotations in annotation_set.items():
        c_vars = {}
        for item_id, annotation in annotations.items():
            try:
                _, _, _, c_var = extract_tempo_from_beats(annotation)
                c_vars[item_id] = c_var
            except Exception as e:
                logger.error('Failed to extract normalized '
                             'tempo std from annotation {}: {}'
                             .format(annotation, e))
        result[version] = c_vars
    return result


def fraction_lt_c_var(norm_c_vars: Dict[str, Dict[str, float]],
                      thresholds: Iterable[float] = None) -> Dict[str, List[float]]:
    """
    Fraction of coefficient of variation values below a threshold (lt == less than).

    :param norm_c_vars: dict with coefficient of variation values per named set
    :param thresholds: list of thresholds
    :return: dict with set name as key and list of fractions as value
    """
    if thresholds is None:
        thresholds = np.arange(0, 0.5, 0.005)
    return {version: fraction_lt_thresholds(norm_c_vars[version].values(), thresholds) for version in norm_c_vars.keys()}


def items_lt_c_var(beat_annotations: Annotations,
                   thresholds: Iterable[float] = None) -> Dict[str, List[Set[str]]]:
    """
    Find item ids of those items with a coefficient of variation
    below a given threshold. If no thresholds are specified, the following
    expression is used::

    thresholds = np.arange(0, 0.5, 0.005)

    :param beat_annotations: dict of annotations with annotation name as key
    :param thresholds: list of coefficient of variation-thresholds
    :return: dict with set names as keys and item lists as values
    """
    if thresholds is None:
        thresholds = np.arange(0, 0.5, 0.005)
    result = {}
    for version, annotations in beat_annotations.items():
        c_vars = []
        for item_id, annotation in annotations.items():
            try:
                _, _, _, c_var = extract_tempo_from_beats(annotation)
                c_vars.append((item_id, c_var))
            except:
                logger.error('Failed to extract normalized tempo '
                             'std from annotation {}'.format(annotation))
        result[version] = [set([v[0]
                                for v in c_vars
                                if v[1] < threshold])
                           for threshold in thresholds]
    return result


def items_in_tempo_intervals(tempo_annotation_set: Annotations,
                             intervals: Iterable[Tuple[int, int]] = None) -> Dict[str, List[Set[str]]]:
    """
    Find item ids with tempo values in given intervals.
    If no intervals are specified, the following expression is used::

    intervals = [(s, s + 11) for s in range(0, 289)]

    :param tempo_annotation_set: dict of annotations with annotation name as key
    :param intervals: list of intervals, given as tuples
    :return: dict with set names as keys and item lists as values
    """
    if intervals is None:
        intervals = [(s, s + 11) for s in range(0, 289)]
    result = {}
    for version, annotations in tempo_annotation_set.items():
        result[version] = []

        # create sorted list of tuples
        tempo_items = [(extract_tempo(annotation), item_id)
                       for item_id, annotation in annotations.items()]
        sorted_tempo_items = sorted(tempo_items, key=lambda x: x[0])

        last_lo = 0
        for interval in intervals:

            # naive implementation:

            # items = []
            # for item, tempo in sorted_item_tempo:
            #     if interval[0] <= tempo <= interval[1]:
            #         items.append(item)
            # result[version].append(items)

            items = []
            pos = bisect_left(sorted_tempo_items, (interval[0],), last_lo)
            last_lo = pos
            for tempo_item in sorted_tempo_items[pos:]:
                if tempo_item[0] <= interval[1]:
                    items.append(tempo_item[1])
                else:
                    break
            result[version].append(set(items))

    return result


def items_per_tag(tag_reference_set: Annotations) -> Dict[str, Dict[str, Set[str]]]:
    """
    Find list of item ids per tag.

    :param tag_reference_set: tag reference set
    :return: nested dicts with annotations name and tag name as
        keys and list of item ids as values
    :rtype: dict[str, dict[str, set[str]]]
    """
    result = {}
    for version, annotations in tag_reference_set.items():
        result[version] = {}
        for item_id, annotation in annotations.items():
            tags = extract_tags(annotation)
            for tag in tags:
                if tag in result[version]:
                    result[version][tag] = result[version][tag].union({item_id})
                else:
                    result[version][tag] = {item_id}
    return result


def fraction_lt_thresholds(values: Iterable[float],
                           thresholds: Iterable[float],
                           key: Callable[[Any], Any] = None) -> List[float]:
    """
    Calculates what fraction of items is less than a given
    threshold for a list of thresholds.

    :param values: values
    :param thresholds: iterable of thresholds
    :param key: key function to extract the attribute to sort by
    :return: list of fractions corresponding to the given thresholds
    :rtype: list[float]
    """
    values_list = list(values)
    sorted_list = sorted(values_list, key=key)
    result = []
    for threshold in thresholds:
        i = bisect_left(sorted_list, threshold)
        fraction = float(i)/len(values_list)
        result.append(fraction)
    return result


def basic_statistics(tempo_annotation_set: Annotations,
                     estimates: bool = False) -> pd.DataFrame:
    """
    Calculate basic statistics about a set of annotations, incl.
    number of tracks, min BPM value, max BPM value, sweet octave start
    (:py:func:`~tempo_eval.evaluation.sweet_octave`), etc.

    :param tempo_annotation_set: set of tempo annotations
    :param estimates: boolean flag indicating whether the annotations are
        estimates or reference annotations
    :return: a DataFrame with the desired values
    :rtype: pandas.DataFrame
    """
    values = {
        'Size': [],
        'Min': [],
        'Max': [],
        'Avg': [],
        'Stdev': [],
        'Sweet Oct. Start': [],
        'Sweet Oct. Coverage': [],
    }
    index = list(sorted(tempo_annotation_set.keys()))
    for version in index:
        annotations = tempo_annotation_set[version]
        # ignore 0.0 BPM estimates.
        tempi = [extract_tempo(annotation)
                 for annotation in annotations.values()
                 if extract_tempo(annotation) > 0.]
        octave, percentage = sweet_octave(tempi)
        avg = mean(tempi)
        if len(tempi) > 1:
            sdev = stdev(tempi, xbar=avg)
        else:
            sdev = np.nan
        values['Size'].append(len(tempi))
        values['Min'].append(float(min(tempi)))
        values['Max'].append(float(max(tempi)))
        values['Avg'].append(float(avg))
        values['Stdev'].append(float(sdev))
        values['Sweet Oct. Start'].append(float(octave))
        values['Sweet Oct. Coverage'].append(float(percentage))

    values_df = pd.DataFrame(values, index=index)
    values_df.name = 'Basic Statistics'
    if estimates:
        values_df.index.name = 'Estimator'
    else:
        values_df.index.name = 'Reference'
    return values_df


@njit
def sweet_octave(tempi: List[float]) -> Tuple[float, float]:
    """
    Calculate the *sweet octave*, i.e. the tempo interval ``[j,2j)`` that contains
    more of the dataset’s songs than any other octave, and its coverage of the dataset.
    If more than one such interval exists, the one with the lowest ``j`` is returned.
    Infinity and NaN values are ignored when calculatig the *sweet octave*, but
    are taken into consideration when calculating coverage.

    .. seealso:: Hendrik Schreiber, Meinard Müller. `A Post-Processing Procedure for
        Improving Music Tempo Estimates Using Supervised Learning.
        <https://www.audiolabs-erlangen.de/content/05-fau/professor/00-mueller/03-publications/2017_SchreiberM_TempoEstimation_ISMIR.pdf>`_
        In Proceedings of the 18th International Society for Music Information
        Retrieval Conference (ISMIR), pages 235–242, Suzhou, China, October 2017.

    :param tempi: list of tempi
    :return: sweet octave start, dataset coverage
    :rtype: (float, float)
    """
    max_in_octave = 0
    max_octave = 0
    for octave_start in range(1, math.ceil(max([v for v in tempi
                                                if not math.isnan(v)
                                                   and not math.isinf(v) and not math.isinf(v)]))):
        in_octave = len([t for t in tempi if octave_start <= t < 2 * octave_start])
        if in_octave > max_in_octave:
            max_in_octave = in_octave
            max_octave = octave_start
    return max_octave, max_in_octave / float(len(tempi))


ACC1 = Metric('Accuracy1',
              formatted_name='Accuracy<sub>1</sub>',
              description='Accuracy<sub>1</sub> is defined as the percentage '
                          'of correct estimates, allowing a 4% tolerance for '
                          'individual BPM values.',
              suitability_function=is_single_bpm,
              eval_function=equal1,
              extract_function=extract_tempo,
              significant_difference_function=mcnemar)
"""Accuracy 1."""

ACC2 = Metric('Accuracy2',
              formatted_name='Accuracy<sub>2</sub>',
              description='Accuracy<sub>2</sub> additionally permits '
                          'estimates to be wrong by a factor of 2, 3, '
                          '1/2 or 1/3 '
                          '(so-called *octave errors*).',
              suitability_function=is_single_bpm,
              eval_function=equal2,
              extract_function=extract_tempo,
              significant_difference_function=mcnemar)
"""Accuracy 2."""

APE1 = Metric('APE1', formatted_name='APE<sub>1</sub>',
              description='APE<sub>1</sub> is defined as absolute '
                          'percentage error between an estimate '
                          'and a reference value: '
                          '<code>APE<sub>1</sub>(E) = |(E-R)/R|</code>.',
              best_value=0.,
              suitability_function=is_single_bpm,
              eval_function=ape1,
              extract_function=extract_tempo,
              significant_difference_function=ttest)
"""Absolute percentage error 1 (APE1)."""

APE2 = Metric('APE2', formatted_name='APE<sub>2</sub>',
              description='APE<sub>2</sub> is the minimum of '
                          'APE<sub>1</sub> allowing the octave '
                          'errors 2, 3, 1/2, and 1/3: '
                          '<code>APE<sub>2</sub>(E) = min('
                          'APE<sub>1</sub>(E), APE<sub>1</sub>(2E), '
                          'APE<sub>1</sub>(3E), APE<sub>1</sub>(&frac12;E), '
                          'APE<sub>1</sub>(&frac13;E))</code>.',
              best_value=0.,
              suitability_function=is_single_bpm,
              eval_function=ape2,
              extract_function=extract_tempo,
              significant_difference_function=ttest)
"""Absolute percentage error 2 (APE2), allowing octave errors."""

PE1 = Metric('PE1', formatted_name='PE<sub>1</sub>',
             description='PE<sub>1</sub> is defined as percentage '
                         'error between an estimate <code>E</code> and a '
                         'reference value <code>R</code>: '
                         '<code>PE<sub>1</sub>(E) = (E-R)/R</code>.',
             best_value=0.,
             suitability_function=is_single_bpm,
             eval_function=pe1,
             signed=True,
             extract_function=extract_tempo,
             significant_difference_function=ttest)
"""Percentage error 1 (PE1)."""

PE2 = Metric('PE2', formatted_name='PE<sub>2</sub>',
             description='PE<sub>2</sub> is the signed PE<sub>1</sub> '
                         'corresponding to the minimum absolute '
                         'PE<sub>1</sub> allowing the octave'
                         'errors 2, 3, 1/2, and 1/3: '
                         '<code>PE<sub>2</sub>(E) = arg min<sub>x</sub>(|x|) with x ∈ '
                         '{PE<sub>1</sub>(E), PE<sub>1</sub>(2E), '
                         'PE<sub>1</sub>(3E), PE<sub>1</sub>(&frac12;E), '
                         'PE<sub>1</sub>(&frac13;E)}</code>',
             best_value=0.,
             suitability_function=is_single_bpm,
             eval_function=pe2,
             signed=True,
             extract_function=extract_tempo,
             significant_difference_function=ttest)
"""Percentage error 2 (PE2), allowing octave errors."""

OE1 = Metric('OE1', formatted_name='OE<sub>1</sub>',
             description='OE<sub>1</sub> is defined as octave '
                         'error between an estimate <code>E</code> and a '
                         'reference value <code>R</code>.'
                         'This means that the most common errors'
                         '&mdash;by a factor of 2 or &frac12;&mdash;'
                         'have the same magnitude, namely 1: '
                         '<code>OE<sub>2</sub>(E) = log<sub>2</sub>(E/R)</code>.',
             best_value=0.,
             unit='TO',
             suitability_function=is_single_bpm,
             eval_function=oe1,
             signed=True,
             extract_function=extract_tempo,
             significant_difference_function=ttest)
"""Octave error 1 (OE1)."""

OE2 = Metric('OE2', formatted_name='OE<sub>2</sub>',
             description='OE<sub>2</sub> is the signed OE<sub>1</sub> '
                         'corresponding to the minimum absolute '
                         'OE<sub>1</sub> allowing the octave'
                         'errors 2, 3, 1/2, and 1/3: '
                         '<code>OE<sub>2</sub>(E) = '
                         'arg min<sub>x</sub>(|x|) with x ∈ '
                         '{OE<sub>1</sub>(E), OE<sub>1</sub>(2E), '
                         'OE<sub>1</sub>(3E), OE<sub>1</sub>(&frac12;E), '
                         'OE<sub>1</sub>(&frac13;E)}</code>',
             best_value=0.,
             unit='TO',
             suitability_function=is_single_bpm,
             eval_function=oe2,
             signed=True,
             extract_function=extract_tempo,
             significant_difference_function=ttest)
"""Octave error 2 (OE2)."""

AOE1 = Metric('AOE1', formatted_name='AOE<sub>1</sub>',
              description='AOE<sub>1</sub> is defined as absolute '
                          'octave error between an estimate '
                          'and a reference value: '
                          '<code>AOE<sub>1</sub>(E) = '
                          '|log<sub>2</sub>(E/R)|</code>.',
              best_value=0.,
              unit='TO',
              suitability_function=is_single_bpm,
              eval_function=aoe1,
              extract_function=extract_tempo,
              significant_difference_function=ttest)
"""Absolute octave error 1 (AOE1)."""

AOE2 = Metric('AOE2', formatted_name='AOE<sub>2</sub>',
              description='AOE<sub>2</sub> is the minimum of '
                          'AOE<sub>1</sub> allowing the octave '
                          'errors 2, 3, 1/2, and 1/3: '
                          '<code>AOE<sub>2</sub>(E) = min('
                          'AOE<sub>1</sub>(E), AOE<sub>1</sub>(2E), '
                          'AOE<sub>1</sub>(3E), AOE<sub>1</sub>(&frac12;E), '
                          'AOE<sub>1</sub>(&frac13;E))</code>.',
              best_value=0.,
              unit='TO',
              suitability_function=is_single_bpm,
              eval_function=aoe2,
              extract_function=extract_tempo,
              significant_difference_function=ttest)
"""Absolute octave error 2 (AOE2)."""

PSCORE = Metric('P-Score', eval_function=p_score,
                description='P-Score is defined as the average of two tempi weighted by '
                            'their perceptual strength, allowing an 8% tolerance for '
                            'both tempo values '
                            '[[MIREX 2006 Definition]'
                            '(https://www.music-ir.org/mirex/wiki/2006:Audio_Tempo_Extraction#Evaluation_Procedures)].',
                unit=None,
                suitability_function=is_mirex_style,
                extract_function=extract_tempi_and_salience,
                significant_difference_function=ttest)
"""P-Score."""

ONE_CORRECT = Metric('One Correct',
                     description='One Correct is the fraction of estimate '
                                 'pairs of which at least one of the two '
                                 'values is equal to a reference value '
                                 '(within an 8% tolerance).',
                     eval_function=one_correct,
                     suitability_function=is_mirex_style,
                     extract_function=extract_tempi_and_salience,
                     unit=None,
                     significant_difference_function=mcnemar)
"""One of two estimates correct."""

BOTH_CORRECT = Metric('Both Correct',
                      description='Both Correct is the fraction of estimate '
                                  'pairs of which both '
                                  'values are equal to the reference values '
                                  '(within an 8% tolerance).',
                      eval_function=both_correct,
                      extract_function=extract_tempi_and_salience,
                      suitability_function=is_mirex_style,
                      unit=None,
                      significant_difference_function=mcnemar)
"""Both correct."""
