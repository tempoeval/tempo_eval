from os.path import join

import pandas as pd
import pytest

from tempo_eval.figures import _find_appropriate_chart_range, render_horizontal_bar_chart, render_line_graph, \
    render_scatter_plot, render_violin_plot


def test_render_horizontal_bar_chart_bad_format():
    df = pd.DataFrame()
    df.name = 'name'
    with pytest.raises(ValueError):
        render_horizontal_bar_chart(['file_name.xxx'], df)


def test_render_line_graph_bad_format():
    df = pd.DataFrame()
    df.name = 'name'
    with pytest.raises(ValueError):
        render_line_graph(['file_name.xxx'], df)


def test_render_scatter_plot_bad_format():
    df = pd.DataFrame()
    df.name = 'name'
    with pytest.raises(ValueError):
        render_scatter_plot(['file_name.xxx'], df)


def test_render_violin_plot_no_content(tmpdir, caplog):
    df = pd.DataFrame({'col': [float('nan')]})
    df.name = 'name'
    render_violin_plot([join(str(tmpdir), 'file_name.png')], df)
    assert 'Violin plot: Dataset without finite values' in caplog.text


def test_find_appropriate_chart_range():
    df = pd.DataFrame({'col': [0.2]})
    chart_range = _find_appropriate_chart_range(df)
    assert chart_range == (0.0, 0.2)

    df = pd.DataFrame({'col': [99.]})
    chart_range = _find_appropriate_chart_range(df)
    assert chart_range == (0.0, 100.)

    df = pd.DataFrame({'col': [0.004]})
    chart_range = _find_appropriate_chart_range(df)
    assert chart_range == (0.0, 0.005)

    # no columns
    df = pd.DataFrame({})
    chart_range = _find_appropriate_chart_range(df)
    assert chart_range == (0.0, 1.0)

    # no values
    df = pd.DataFrame({'col': []})
    chart_range = _find_appropriate_chart_range(df)
    assert chart_range == (0.0, 1.0)

    # negative values
    df = pd.DataFrame({'col': [0.2, -0.1]})
    chart_range = _find_appropriate_chart_range(df)
    assert chart_range == (-0.1, 0.2)
