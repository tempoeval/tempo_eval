"""
Rudimentary support for writing markdown.
"""
from typing import Any, Dict, Callable

import pandas as pd


class MarkdownWriter:
    """
    Simple wrapper around a file to offer markdown writing support.
    """

    def __init__(self, file, jekyll: bool = True, kramdown: bool = True) -> None:
        """
        Wrap file in writer.

        :param kramdown: use `kramdown <https://kramdown.gettalong.org>`_ dialect
            (Jekyll's default Markdown renderer)
        :param jekyll: render for Jekyll
        :param file: file
        """
        super().__init__()
        self.file = file
        self.figure_count = 0
        self.table_count = 0
        self.front_matter_written = False
        self.kramdown = kramdown
        self.jekyll = jekyll

    def write(self, *objects: Any, **kwargs) -> None:
        """
        Write an object, possibly with some formatting.

        :param objects: array of objects
        :param kwargs: possible arguments are ``emphasize=True``, ``strong=True``,
            or ``target='http://www.something.org'``
        """
        if not self.front_matter_written and self.jekyll:
            self.front_matter(None)
        for i, o in enumerate(objects):
            s = str(o)
            if kwargs is not None:
                if 'emphasize' in kwargs and kwargs['emphasize']:
                    s = '*{}*'.format(s.replace('*', '\\*'))
                if 'strong' in kwargs and kwargs['strong']:
                    s = '__{}__'.format(s.replace('_', '\\_'))
                if 'target' in kwargs:
                    s = '[{}]({})'.format(s, kwargs['target'])
            if i < len(objects) - 1:
                s += ' '
            self.file.write(s)

    def writeln(self, *objects: Any, **kwargs) -> None:
        """
        Write some objects, followed by a newline (has no effect in Markdown).

        :param objects: array of objects
        :param kwargs: possible arguments are ``emphasize=True``, ``strong=True``,
            or ``target='http://www.something.org'``
        """
        for o in objects:
            self.write(str(o), **kwargs)
        self.write('\n')

    def paragraph(self, *objects: Any, **kwargs) -> None:
        """
        Write a paragraph, i.e., some text followed by two newlines.

        :param objects: array of objects
        :param kwargs: possible arguments are ``emphasize=True``, ``strong=True``,
            or ``target='http://www.something.org'``
        """
        for o in objects:
            self.write(str(o), **kwargs)
        self.write('\n\n')

    def headline(self, level: int, headline: Any) -> None:
        """
        Write a headline of a given level.

        :param level: level, must be greater than 0
        :param headline: headline
        """
        if not self.front_matter_written:
            self.front_matter(headline)
        if level < 1:
            raise ValueError('Headline level must be greater than 0: {}'.format(level))
        self.write(('#' * level) + ' ' + headline + '\n\n')

    def front_matter(self, title: Any, layout: Any = 'default') -> None:
        """
        Jekyll requires that Markdown files have `front matter
        <https://jekyllrb.com/docs/front-matter/>`_
        defined at the top of every file.

        :param title: title
        :param layout: layout
        """
        if self.jekyll:
            self.file.write('---\n')
            if title is not None:
                self.file.write('title: {}\n'.format(title))
            if layout is not None:
                self.file.write('layout: {}\n'.format(layout))
            self.file.write('---\n\n')
            self.front_matter_written = True

    def h1(self, headline: Any) -> None:
        """
        Write a level 1 headline.

        :param headline: headline
        """
        self.headline(1, headline)

    def h2(self, headline: Any) -> None:
        """
        Write a level 2 headline.

        :param headline: headline
        """
        self.headline(2, headline)

    def h3(self, headline: Any) -> None:
        """
        Write a level 3 headline.

        :param headline: headline
        """
        self.headline(3, headline)

    def h4(self, headline: Any) -> None:
        """
        Write a level 4 headline.

        :param headline: headline
        """
        self.headline(4, headline)

    def h5(self, headline: Any) -> None:
        """
        Write a level 5 headline.

        :param headline: headline
        """
        self.headline(5, headline)

    def h6(self, headline: Any) -> None:
        """
        Write a level 6 headline.

        :param headline: headline
        """
        self.headline(6, headline)

    def emphasize(self, text: Any) -> None:
        """
        Write some text with emphasis.

        :param text: text
        """
        self.write(text, emphasize=True)

    def strong(self, text: Any) -> None:
        """
        Write some "strong" text.

        :param text: text
        """
        self.write(text, strong=True)

    def blockquote(self, text: Any) -> None:
        """
        Write a blockquote.

        :param text: text
        """
        self.write('> {}\n\n'.format(text))

    def rule(self) -> None:
        """
        Write a horizontal rule.
        """
        self.writeln('-------------------------')

    def link(self, name: Any, target: Any) -> None:
        """
        Write a link pointing to the given target using the given name.

        :param name: link name
        :param target: link target
        """
        self.write(self.to_link(name, target))

    def headline_link(self, headline: Any) -> None:
        """
        Write a headline link pointing to the given headline using its name.

        :param headline: headline name
        """
        return self.write(self.to_headline_link(headline))

    def to_link(self, name: Any, target: Any) -> str:
        """
        Turn the given name and target into a Markdown link.

        :param name: link name
        :param target: link target
        :return: Markdown link
        """
        return '[{}]({})'.format(name, target)

    def to_headline_link(self, headline: Any) -> str:
        """
        Turn the given headline into a Markdown link.

        :param headline: headline name
        :return: Markdown link
        """
        target = '#' + str(headline).lower().strip() \
            .replace(' ', '-') \
            .replace('/', '') \
            .replace('.', '')
        # # for GitHub Markdown
        # # see https://stackoverflow.com/a/38507669/942774
        # target = '#' + str(headline).lower().strip().replace(' ', '-').translate(
        #     str.maketrans(dict.fromkeys(string.punctuation)))

        return self.to_link(headline, target)

    def table(self,
              values_df: pd.DataFrame,
              caption: Any = None,
              strong: Callable[[Any], bool] = None,
              emphasize: Callable[[Any], bool] = None,
              column_strong: Callable[[pd.Series], Any] = None,
              column_emphasize: Callable[[pd.Series], Any] = None,
              float_format: str = ':.4f',
              int_format: str = '',
              string_format: str = '',
              tables: Dict[str, str] = {}) -> None:
        """
        Write a markdown table based on the given ``values_df`` dataframe, including
        download links for tables in several formats (if given) and a caption.

        :param values_df: data as Panda dataframe
        :param caption: caption
        :param strong: function that returns a bool, indicating whether a value should be formatted as strong
        :param emphasize: function that returns a bool, indicating whether a value should be formatted as emphasized
        :param column_strong: function that takes a DataFrame column (:py:class:`~pandas.Series`)
            as argument and returns the value that should be strongly formatted,
            e.g., :py:func:`~pandas.Series.max`.
        :param column_emphasize: function that takes a DataFrame column (:py:class:`~pandas.Series`)
            as argument and returns the value that should be formatted with an emphasis,
            e.g., :py:func:`~pandas.Series.max`.
        :param float_format: formatter string for ``float`` values
        :param int_format: formatter string for ``int`` values
        :param string_format: formatter string for ``string`` values
        :param tables: dict of formats and filenames, e.g. 'csv': 'some_file.csv'
        """

        if values_df.index.name:
            self.write('| ' + values_df.index.name)
        else:
            self.write('|&nbsp;')
        for column in values_df.columns:
            self.write('| {} '.format(column))
        self.writeln('|')

        self.write('| ---: ')
        for column in range(len(values_df.columns)):
            self.write('| :---: ')
        self.writeln('|')

        if column_strong:
            strong_values = [column_strong(values_df[column]) for column in values_df]
        else:
            strong_values = [None] * len(values_df.columns)

        if column_emphasize:
            emphasize_values = [column_emphasize(values_df[column]) for column in values_df]
        else:
            emphasize_values = [None] * len(values_df.columns)

        for row_index, row in enumerate(values_df.itertuples(index=False)):
            index_value = values_df.index.values[row_index]
            self.write('| {:50} '.format(self.to_headline_link(index_value)))
            for col_index, k in enumerate(row):
                b = '  '
                if column_emphasize or column_strong:
                    if emphasize_values[col_index] == k:
                        b = '*'
                    if strong_values[col_index] == k:
                        b = '__'
                else:
                    if strong and strong(k):
                        b = '__'
                    if emphasize and emphasize(k):
                        b = '*'
                if isinstance(k, float):
                    self.write(('| {}{' + float_format + '}{} ').format(b, k, b))
                elif isinstance(k, int):
                    self.write(('| {}{' + int_format + '}{} ').format(b, k, b))
                else:
                    self.write(('| {}{' + string_format + '}{} ').format(b, k, b))
            self.writeln('|')
        self.writeln()

        self.table_count += 1
        if caption:
            self.paragraph('<a name="table{}"></a>Table {}: {}'.format(self.table_count, self.table_count, caption))
        for k, v in tables.items():
            self.write('[{format}]({path} "Download data as {format}") '.format(format=k.upper(), path=v))
        self.paragraph()

    def figure(self,
               tables: Dict[str, str] = {},
               images: Dict[str, str] = {},
               caption: Any = None) -> None:
        """
        Write a figure given various image file names.

        :param tables: dict of data files with extension as key and file name as value, should contain CSV
        :param images: dict of image files with extension as key and file name as value
        :param caption: caption
        """
        file_name = None

        if 'svg' in images:
            file_name = images['svg']
        elif 'pdf' in images:
            file_name = images['pdf']
        elif 'png' in images:
            file_name = images['png']
        elif len(images) > 1:
            file_name = next(images.values())

        if file_name is None:
            raise ValueError('No image specified')

        # <embed> instead of <img> keeps the JavaScript in the SVG active.
        self.paragraph('<figure>\n<embed type="image/svg+xml" src="{file_name}">\n</figure>'
                       .format(file_name=file_name))

        self.figure_count += 1
        if caption:
            self.paragraph('<a name="figure{}"></a>Figure {}: {}'
                           .format(self.figure_count, self.figure_count, caption))

        for k, v in tables.items():
            self.write('[{format}]({path} "Download data as {format}") '.format(format=k.upper(), path=v))
        for k, v in images.items():
            self.write('[{format}]({path} "Open Figure") '.format(format=k.upper(), path=v))
        self.paragraph()
