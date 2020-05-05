"""
Support for figure generation using Pygal and MatPlotLib.
"""
import logging
from math import isfinite
from os.path import splitext
from typing import Tuple, List, Iterable, Dict

import matplotlib as mpl
import numpy as np
import pandas as pd
import pygal
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from pygal.style import Style

PNG_DPI = 300
SUPPORTED_FORMATS = ['.pdf', '.eps', '.png', '.ps', '.svg']
PYGAL_STYLE = Style(
    background='transparent',
    plot_background='white',
    font_family='Helvetica, Arial, sans-serif',
    label_font_size=15,
    major_label_font_size=15,
    title_font_size=18,
    legend_font_size=15,
    value_font_size=15,
    value_label_font_size=15,
    guide_stroke_dasharray=1,
    major_guide_stroke_dasharray=1,
)
# use pygal color cycle from matplotlib, because it simply has more default colors
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=Style.colors)


def render_horizontal_bar_chart(file_names: Iterable[str],
                                values_df: pd.DataFrame,
                                stdev_df: pd.DataFrame = None,
                                caption: str = None,
                                x_axis_label: str = None) -> None:
    """
    Render a horizontal bar chart using the data provided
    in the ``values_df`` dataframe.

    :param file_names: output files
    :param values_df: dataframe of values
    :param stdev_df: dataframe of standard deviations for the given values
    :param caption: caption
    :param x_axis_label: label for the X axis
    """

    _check_format(file_names)

    logging.debug('Rendering horizontal bar chart to {}.'.format(file_names))

    svgs, others = _split_file_names(file_names)
    if svgs:
        _render_horizontal_bar_chart_pygal(svgs, values_df, stdev_df=stdev_df,
                                           caption=caption, x_axis_label=x_axis_label)

    if others:
        _render_horizontal_bar_chart_matplotlib(others, values_df, stdev_df=stdev_df,
                                                caption=caption, x_axis_label=x_axis_label)


def render_line_graph(file_names: Iterable[str],
                      values_df: pd.DataFrame,
                      confidence_interval: bool = False,
                      caption: str = None,
                      y_axis_label: str = None) -> None:
    """
    Render a line graph using the data provided in the
    ``values_df`` dataframe.

    :param file_names: output files
    :param values_df: dataframe of values
    :param confidence_interval: if True, the dataframe is organized so that
        a data column is followed by two confidence interval columns
    :param caption: caption
    :param y_axis_label: label for the Y axis
    """

    _check_format(file_names)

    logging.debug('Rendering line graph to {}.'.format(file_names))

    if confidence_interval:
        _render_linear_gam_plot_matplotlib(file_names, values_df, caption=caption, y_axis_label=y_axis_label)
    else:
        svgs, others = _split_file_names(file_names)
        if svgs:
            _render_line_graph_pygal(svgs, values_df, caption=caption, y_axis_label=y_axis_label)
        if others:
            _render_line_graph_matplotlib(others, values_df, caption=caption, y_axis_label=y_axis_label)


def render_scatter_plot(file_names: Iterable[str],
                        values_df: pd.DataFrame,
                        caption: str = None,
                        y_axis_label: str = None) -> None:
    """
    Render a scatter plot the data provided in the
    ``values_df`` dataframe.

    :param file_names: output files
    :param values_df: dataframe of values
    :param caption: caption
    :param y_axis_label: label for the Y axis
    """

    _check_format(file_names)

    logging.debug('Rendering scatter plot to {}.'.format(file_names))
    _render_scatter_plot_matplotlib(file_names, values_df, caption=caption, y_axis_label=y_axis_label)


def render_violin_plot(file_names: Iterable[str],
                       values_df: pd.DataFrame,
                       caption: str = None,
                       axis_label: str = None) -> None:
    """
    Render a violin plot using the data provided in the
    ``values_df`` dataframe.

    :param file_names: output files
    :param values_df: dataframe of values
    :param caption: caption
    :param axis_label: label for the Y axis
    """

    _check_format(file_names)

    logging.debug('Rendering violin plot to {}.'.format(file_names))
    _render_violin_plot_matplotlib(file_names, values_df, caption=caption, axis_label=axis_label)


def render_tag_violin_plot(file_names: Iterable[str],
                           values_df: Dict[str,pd.DataFrame],
                           caption: str = None,
                           axis_label: str = None) -> None:
    """
    Render tag violin plots using the data provided in the
    ``values_df`` dataframe dict.

    :param file_names: output files
    :param values_df: dict of dataframes with (grouping) tags as key
    :param caption: caption
    :param axis_label: label for the Y axis
    """

    _check_format(file_names)

    logging.debug('Rendering tag violin plot to {}.'.format(file_names))
    _render_tag_violin_plot_matplotlib(file_names, values_df, caption=caption, axis_label=axis_label)


def _render_horizontal_bar_chart_pygal(file_names: Iterable[str],
                                       values_df: pd.DataFrame,
                                       stdev_df: pd.DataFrame = None,
                                       caption: str = None,
                                       x_axis_label: str = None) -> None:

    convert = lambda d: d
    chart_range = _find_appropriate_chart_range(values_df)
    formatter = lambda x: "%.2f" % x
    if x_axis_label is not None and '%' in x_axis_label:
        convert = lambda d: d * 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)
        formatter = lambda x: "%.1f" % x

    bar_chart = pygal.HorizontalBar(range=chart_range)
    try:
        bar_chart.title = values_df.name
    except AttributeError:
        bar_chart.title = 'No Name'
    bar_chart.x_title = x_axis_label
    bar_chart.truncate_legend = -1
    bar_chart.legend_at_bottom = True
    bar_chart.legend_at_bottom_columns = 2
    bar_chart.style = PYGAL_STYLE
    bar_chart.value_formatter = formatter
    bar_chart.show_y_guides = True
    bar_chart.show_x_guides = True
    df_length = len(values_df)

    if len(values_df.columns) > 1:
        bar_chart.margin_bottom = 50 + (df_length // 2) * 5  # make space for legend (based on heuristic)
        bar_chart.width = 800
        bar_chart.height = 600 + (df_length // 2) * 5 * len(values_df.columns)  # make sure this does not get squashed
        bar_chart.x_labels = map(str, values_df.columns)

        for index, row in values_df.iterrows():
            bar_chart.add(index, list(map(convert, row)))

        # for k, v in values_df.items():
        #     bar_chart.add(k, list(map(convert, v)))
    else:
        bar_chart.width = 800
        bar_chart.height = 600 + df_length * 5  # make sure this does not get squashed
        bar_chart.add(None, list(map(convert, reversed(values_df[values_df.columns[0]].values))))
        bar_chart.x_labels = map(str, reversed(values_df.index.values))

    for f in file_names:
        bar_chart.render_to_file(f)


def _render_horizontal_bar_chart_matplotlib(file_names: Iterable[str],
                                            values_df: pd.DataFrame,
                                            stdev_df: pd.DataFrame = None,
                                            caption: str = None,
                                            x_axis_label: str = None) -> None:

    convert = lambda d: d
    chart_range = _find_appropriate_chart_range(values_df)
    if x_axis_label is not None and '%' in x_axis_label:
        convert = lambda d: d * 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)
    df_length = len(values_df)

    fig = plt.figure()
    if caption:
        fig.canvas.set_window_title(caption)
    ax = fig.add_subplot(111)
    ax.set_title(values_df.name)
    ax.set_xlabel(x_axis_label)
    ax.grid(True, axis='x')
    ax.set_xlim(chart_range[0], chart_range[1])

    if len(values_df.columns) > 1:
        fig.set_size_inches(7, 10 + len(values_df.columns) * 0.075)
        height = 0.8 / df_length
        ind = np.arange(len(values_df.columns))
        i = 0
        for index, row in values_df.iterrows():
            stdev = None
            # stdev disabled for now
            # if stdev_df is not None:
            #     stdev = list(map(convert, stdev_dictionary[k]))
            ax.barh(ind + i * height, list(map(convert, row)), height=height, label=index, xerr=stdev)
            i += 1
        plt.yticks(ind + 0.4, values_df.columns)
        ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.05), ncol=2, frameon=False)
    else:
        fig.set_size_inches(7, 5 + df_length * 0.075)
        ind = np.arange(df_length)
        values = list(map(convert, values_df[values_df.columns[0]].values))
        stdev = None
        # stdev disabled for now
        # if stdev_df is not None:
        #     stdev = list(map(convert, values_df[stdev_df.columns[0]].values))
        ax.barh(ind, values, xerr=stdev)
        plt.yticks(ind, values_df.index.values)
        ax.invert_yaxis()

    _save_matplotlib(fig, file_names)


def _render_line_graph_pygal(file_names: Iterable[str],
                             values_df: pd.DataFrame,
                             caption: str = None,
                             y_axis_label: str = None) -> None:

    # Pygal apparently does not like NaNs so much.
    # We deal with them by dropping all rows containing them.
    values_df_na_dropped = values_df.dropna()

    y_convert = lambda d: d
    chart_range = _find_appropriate_chart_range(values_df_na_dropped)
    formatter = lambda x: "%.4f" % x
    if y_axis_label and '%' in y_axis_label:
        y_convert = lambda d: d * 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)
        formatter = lambda x: "%.1f" % x
    x_convert = lambda d: str(d)
    if '%' in values_df.index.name:
        x_convert = lambda d: str(round(d * 100., 4))

    # make space for legend (based on heuristic)
    height = 600
    if len(values_df_na_dropped.columns) > 8:
        height = height + (len(values_df_na_dropped.columns)-8) * 12

    line_chart = pygal.Line(show_only_major_dots=True, range=chart_range,
                            legend_at_bottom=True, height=height)
    line_chart.truncate_legend = -1

    # make space for legend (based on heuristic)
    line_chart.margin_bottom = 50 + (len(values_df_na_dropped.columns) // 2) * 5

    line_chart.legend_at_bottom_columns = 2
    line_chart.style = PYGAL_STYLE
    try:
        line_chart.title = values_df.name
    except AttributeError:
        line_chart.title = 'No Name'
    line_chart.y_title = y_axis_label
    line_chart.x_title = values_df.index.name
    line_chart.x_labels = list(map(x_convert, values_df_na_dropped.index.values))
    line_chart.x_label_rotation = 90
    line_chart.show_minor_x_labels = False
    line_chart.x_labels_major_count = 11
    line_chart.show_x_guides = True
    line_chart.value_formatter = formatter

    for column in values_df_na_dropped.columns:
        values = list(map(y_convert, values_df_na_dropped[column].values))
        line_chart.add(column, values)

    for f in file_names:
        line_chart.render_to_file(f)


def _render_scatter_plot_pygal(file_names: Iterable[str],
                               values_df: pd.DataFrame,
                               caption: str = None,
                               y_axis_label: str = None) -> None:

    y_convert = lambda d: d
    chart_range = _find_appropriate_chart_range(values_df)
    formatter = lambda x: "%.2f" % x
    if y_axis_label and '%' in y_axis_label:
        y_convert = lambda d: d * 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)
        formatter = lambda x: "%.1f" % x
    x_convert = lambda d: d
    if '%' in values_df.index.name:
        x_convert = lambda d: round(d * 100.)

    scatter_plot = pygal.XY(range=chart_range, legend_at_bottom=True, stroke=False, dots_size=2)
    scatter_plot.truncate_legend = -1
    scatter_plot.margin_bottom = 50 + (len(values_df.columns) // 2) * 5  # make space for legend (based on heuristic)
    scatter_plot.legend_at_bottom_columns = 2
    scatter_plot.style = PYGAL_STYLE
    try:
        scatter_plot.title = values_df.name
    except AttributeError:
        scatter_plot.title = 'No Name'
    scatter_plot.y_title = y_axis_label
    scatter_plot.x_title = values_df.index.name
    scatter_plot.x_label_rotation = 90
    scatter_plot.show_minor_x_labels = False
    scatter_plot.x_labels_major_count = 11
    scatter_plot.show_x_guides = True
    scatter_plot.value_formatter = formatter

    for column in values_df.columns:
        pairs = [(x_convert(x), y)
                 for x, y in zip(values_df.index, map(y_convert, values_df[column].values))
                 if isfinite(y) and isfinite(x)]
        scatter_plot.add(column, pairs)

    for f in file_names:
        scatter_plot.render_to_file(f)


def _render_box_plot_pygal(file_names: Iterable[str],
                           values_df: pd.DataFrame,
                           caption: str = None,
                           y_axis_label: str = None) -> None:

    y_convert = lambda d: d
    chart_range = _find_appropriate_chart_range(values_df)
    formatter = lambda x: "%.2f" % x
    if y_axis_label and '%' in y_axis_label:
        y_convert = lambda d: d * 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)
        formatter = lambda x: "%.1f" % x
    x_convert = lambda d: d
    if '%' in values_df.index.name:
        x_convert = lambda d: round(d * 100.)

    box_plot = pygal.Box(range=chart_range, legend_at_bottom=True, mode="stdev")
    box_plot.truncate_legend = -1
    box_plot.margin_bottom = 50 + (len(values_df.columns) // 2) * 5  # make space for legend (based on heuristic)
    box_plot.legend_at_bottom_columns = 2
    box_plot.style = PYGAL_STYLE
    try:
        box_plot.title = values_df.name
    except AttributeError:
        box_plot.title = 'No Name'
    box_plot.y_title = y_axis_label
    box_plot.x_title = values_df.index.name
    #box_plot.x_label_rotation = 90
    #box_plot.show_minor_x_labels = False
    #box_plot.x_labels_major_count = 11
    #box_plot.show_x_guides = True
    box_plot.value_formatter = formatter

    for column in values_df.columns:
        values = list(map(y_convert, values_df[column].values))
        box_plot.add(column, values)

    for f in file_names:
        box_plot.render_to_file(f)


def _render_line_graph_matplotlib(file_names: Iterable[str],
                                  values_df: pd.DataFrame,
                                  caption: str = None,
                                  y_axis_label: str = None) -> None:
    y_convert = lambda d: d
    chart_range = _find_appropriate_chart_range(values_df)
    if y_axis_label and '%' in y_axis_label:
        y_convert = lambda d: d * 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)

    x_convert = lambda d: d
    if '%' in values_df.index.name:
        x_convert = lambda d: d * 100.

    fig = plt.figure()
    if caption:
        fig.canvas.set_window_title(caption)
    fig.set_size_inches(7, 5 + (len(values_df.columns) // 2) * 0.2)  # allow extra space for legend
    ax = fig.add_subplot(111)
    ax.set_xlabel(values_df.index.name)
    ax.set_ylabel(y_axis_label)
    ax.set_title(values_df.name)
    ax.grid(True)
    ax.set_ylim(chart_range[0], chart_range[1])
    x = list(map(x_convert, values_df.index.values))
    for column in values_df.columns:
        l = list(map(y_convert, values_df[column].values))
        ax.plot(x, l, label=column)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

    _save_matplotlib(fig, file_names)


def _render_scatter_plot_matplotlib(file_names: Iterable[str],
                                    values_df: pd.DataFrame,
                                    caption: str = None,
                                    y_axis_label: str = None) -> None:
    y_convert = lambda d: d
    chart_range = _find_appropriate_chart_range(values_df)
    if y_axis_label and '%' in y_axis_label:
        y_convert = lambda d: d * 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)

    fig = plt.figure()
    if caption:
        fig.canvas.set_window_title(caption)
    fig.set_size_inches(7, 5 + (len(values_df.columns) // 2) * 0.2)  # allow extra space for legend
    ax = fig.add_subplot(111)
    ax.set_xlabel(values_df.index.name)
    ax.set_ylabel(y_axis_label)
    ax.set_title(values_df.name)
    ax.grid(b=True, which='major', linestyle='-')
    ax.grid(b=True, which='minor', linestyle='--')
    ax.set_ylim(chart_range[0], chart_range[1])
    # ax.set_xscale('log')
    # # Rewrite the x labels
    # formatter = ScalarFormatter()
    # formatter.set_scientific(False)
    # ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_minor_formatter(formatter)
    for column in values_df.columns:
        Xy = np.array([[v, d]
                       for v, d in zip(values_df.index.values, values_df[column].values)
                       if isfinite(d) and isfinite(v)])


        X = Xy[:,0]
        y = Xy[:,1]
        ax.scatter(X, y, s=2, label=column)

        # add pygam

        # X = np.expand_dims(X, axis=1)
        # gam = LinearGAM().gridsearch(X, y)
        # XX = gam.generate_X_grid(term=0, n=100)
        # # ax.plot(XX, gam.predict(XX), 'r--', label=column + 'GAM')
        # # ax.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--', label=column + 'GAM_CI')
        # ax.plot(XX, gam.predict(XX), label=column)
        # ax.plot(XX, gam.prediction_intervals(XX, width=.5), color='lightgrey', ls='dotted')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

    _save_matplotlib(fig, file_names)


def _render_scatter_plot_matplotlib(file_names: Iterable[str],
                                    values_df: pd.DataFrame,
                                    caption: str = None,
                                    y_axis_label: str = None) -> None:
    factor = 1.
    chart_range = _find_appropriate_chart_range(values_df)
    if y_axis_label and '%' in y_axis_label:
        factor = 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)

    fig = plt.figure()
    if caption:
        fig.canvas.set_window_title(caption)
    fig.set_size_inches(7, 5 + (len(values_df.columns) // 2) * 0.2)  # allow extra space for legend
    ax = fig.add_subplot(111)
    ax.set_xlabel(values_df.index.name)
    ax.set_ylabel(y_axis_label)
    ax.set_title(values_df.name)
    ax.grid(b=True, which='major', linestyle='-')
    ax.grid(b=True, which='minor', linestyle='--')
    ax.set_ylim(chart_range[0], chart_range[1])
    for column in values_df.columns:
        Xy = np.array([[v, d * factor]
                       for v, d in zip(values_df.index.values, values_df[column].values)
                       if isfinite(d) and isfinite(v)])
        X = Xy[:,0]
        y = Xy[:,1]
        ax.scatter(X, y, s=2, label=column)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

    _save_matplotlib(fig, file_names)


def _render_linear_gam_plot_matplotlib(file_names: Iterable[str],
                                       values_df: pd.DataFrame,
                                       caption: str = None,
                                       y_axis_label: str = None) -> None:
    factor = 1.
    chart_range = _find_appropriate_chart_range(values_df)
    if y_axis_label and '%' in y_axis_label:
        factor = 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)

    axes_num = len(values_df.columns) // 3
    fig, axes = plt.subplots(nrows=axes_num, ncols=1, sharex=True, squeeze=False)  # sharey=True

    # add a big axes, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none',
                    top=False, bottom=False,
                    left=False, right=False,
                    which='both')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(values_df.name)

    if caption:
        fig.canvas.set_window_title(caption)
    fig.set_size_inches(7, 2 + len(values_df.columns) * 0.23)
    fig.text(0.98, 0.5, y_axis_label, ha='right',
             va='center', rotation='vertical')

    for i, ax in enumerate(axes[:, 0]):
        column = values_df.columns[i*3]
        column_lower_conf = values_df.columns[i*3+1]
        column_upper_conf = values_df.columns[i*3+2]

        if (i+1)*3 >= len(values_df.columns):
            ax.set_xlabel(values_df.index.name)

        ax.set_ylabel(column, rotation='horizontal', ha='right')
        ax.grid(b=True, which='major', linestyle='-')
        ax.grid(b=True, which='minor', linestyle='--')
        ax.set_ylim(chart_range[0], chart_range[1])

        ax.fill_between(values_df.index.values,
                        values_df[column_lower_conf].values * factor,
                        values_df[column_upper_conf].values * factor,
                        color='lightgrey')
        ax.plot(values_df.index.values,
                values_df[column].values * factor,
                color='r')

    _save_matplotlib(fig, file_names)


def _render_violin_plot_matplotlib(file_names: Iterable[str],
                                   values_df: pd.DataFrame,
                                   caption: str = None,
                                   axis_label: str = None) -> None:
    convert = lambda d: d
    chart_range = _find_appropriate_chart_range(values_df)
    if axis_label and '%' in axis_label:
        convert = lambda d: d * 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)

    fig = plt.figure()
    if caption:
        fig.canvas.set_window_title(caption)
    fig.set_size_inches(7, 5 + (len(values_df.columns) // 2) * 0.2)  # allow extra space for legend
    ax = fig.add_subplot(111)
    ax.grid(b=True)
    ax.set_axisbelow(True)
    ax.set_ylabel(values_df.index.name)
    ax.set_xlabel(axis_label)
    ax.set_title(values_df.name)
    ax.set_xlim(chart_range[0], chart_range[1])

    data = []
    for column in reversed(values_df.columns):
        sample = [d for d in map(convert, values_df[column].values) if isfinite(d)]
        if sample:
            data.append(sample)
        else:
            logging.warning('Violin plot: Dataset without finite values: {}. Using single zero distribution.'.format(column))
            data.append([0.])
    # ax.boxplot(data, showmeans=True, meanline=True)
    parts = ax.violinplot(data, showmeans=True, showextrema=False, vert=False)

    # customize colors
    parts['cmeans'].set_color('black')
    for pc in parts['bodies']:
        pc.set_facecolor('red')
        pc.set_alpha(0.75)

    plt.yticks(list(range(1, len(values_df.columns) + 1)), list(reversed(values_df.columns)))
    _save_matplotlib(fig, file_names)


def _render_tag_violin_plot_matplotlib(file_names: Iterable[str],
                                   values_df: Dict[str, pd.DataFrame],
                                   caption: str = None,
                                   axis_label: str = None) -> None:
    tags = list(values_df.keys())
    tags.sort()
    one_df = values_df[tags[0]]

    convert = lambda d: d
    chart_range = np.inf, -np.inf
    for df in values_df.values():
        lo, hi = _find_appropriate_chart_range(df)
        chart_range = min(chart_range[0], lo), max(chart_range[1], hi)

    if axis_label and '%' in axis_label:
        convert = lambda d: d * 100.
        chart_range = (chart_range[0] * 100.0, chart_range[1] * 100.0)

    fig = plt.figure()
    if caption:
        fig.canvas.set_window_title(caption)
    fig.set_size_inches(7, 5 + len(one_df.columns) * 0.2)  # allow extra space for legend
    ax = fig.add_subplot(111)
    ax.grid(b=True, axis='x')
    ax.set_axisbelow(True)
    ax.set_ylabel(one_df.index.name)
    ax.set_xlabel(axis_label)
    ax.set_title(one_df.name)
    ax.set_xlim(chart_range[0], chart_range[1])

    # our own fake grid
    for i in range(len(tags)+1):
        ax.axhline(i+0.5, color='lightgray', lw=0.5)

    data = []
    positions = []
    colors = []
    pos = 0
    patches = []
    legend_labels = []
    for tag in tags:
        df = values_df[tag]
        for i, column in enumerate(df.columns):
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            colors.append(color)
            if len(patches) < len(df.columns):
                patches.append(Patch(color=color))
                legend_labels.append(column)

            sample = [d for d in map(convert, df[column].values) if isfinite(d)]
            if sample:
                data.append(sample)
            else:
                logging.warning('Violin plot: Dataset without finite values: {}. Using single zero distribution.'.format(column))
                data.append([0.])
            positions.append(pos)
            pos += 1
        pos += 2
    # ax.boxplot(data, showmeans=True, meanline=True)
    slots = (len(one_df.columns) + 2.)
    width = 1. / slots
    positions = np.flip(np.array(positions) / slots + 0.5 + width)
    parts = ax.violinplot(data, positions=positions,
                          widths=width,
                          showmeans=True,
                          showextrema=False, vert=False)

    # customize colors per algorithm
    parts['cmeans'].set_color('black')
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.75)

    lgd = ax.legend(patches, legend_labels, loc='upper center',
                    bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.yticks(list(range(1, len(tags) + 1)), list(reversed(tags)))
    _save_matplotlib(fig, file_names, bbox_extra_artists=(lgd,))


def _find_appropriate_chart_range(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Attempt to find an appropriate value range, so that all values in the dataframe's
    columns can be displayed.
    In case the dataframe does not contain any columns or values, this function returns
    ``(0., 1.)``.

    :param df: dataframe
    :return: appropriate value range
    """
    if df.dropna().size == 0:
        return 0., 1.

    v = [1., 2., 3., 5.]

    min_value = df.dropna().values.min()
    lo = 0.
    if min_value < 0.:
        f = -100000
        searching_lo = True
        while searching_lo:
            for i in v:
                lo = i / f
                if min_value >= lo:
                    searching_lo = False
                    break
            f /= 10

    max_value = df.dropna().values.max()
    hi = 0.
    if max_value > 0.:
        f = 100000
        searching_hi = True
        while searching_hi:
            for i in v:
                hi = i / f
                if max_value <= hi:
                    searching_hi = False
                    break
            f /= 10

    return lo, hi


def _save_matplotlib(fig, file_names: Iterable[str], bbox_extra_artists=None) -> None:
    """
    Save figure to one or more files.

    :param bbox_extra_artists: extra artists to make space for (e.g. legend)
    :param fig: figure
    :param file_names: one or multiple filenames
    """

    for f in file_names:
        dpi = None
        if f.lower().endswith('.png'):
            dpi = PNG_DPI
        fig.savefig(f, dpi=dpi,
                    bbox_extra_artists=bbox_extra_artists,
                    bbox_inches='tight')

    plt.close()


def _split_file_names(file_names: Iterable[str]) -> Tuple[List[str], List[str]]:
    svgs = []
    others = []
    for f in file_names:
        if f.lower().endswith('.svg'):
            svgs.append(f)
        else:
            others.append(f)
    return svgs, others


def _check_format(file_names: Iterable[str]):
    for f in file_names:
        extension = splitext(f)[1]
        if extension not in SUPPORTED_FORMATS:
            raise ValueError('Unsupported output format: {}'.format(extension))

