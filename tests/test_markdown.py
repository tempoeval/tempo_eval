from os import remove
import tempfile

import pytest

from tempo_eval.markdown_writer import MarkdownWriter


def test_write_multiple_objects(tmpdir):
    try:

        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8',
                                         delete=False) as w:
            md = MarkdownWriter(w, jekyll=False, kramdown=False)
            md.write('1', '2', '3')

        with open(w.name, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert content == '1 2 3'

    finally:
        remove(w.name)


def test_write_target(tmpdir):
    try:

        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8',
                                         delete=False) as w:
            md = MarkdownWriter(w, jekyll=False, kramdown=False)
            md.write('name', target='http://www.some.com/')

        with open(w.name, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert content == '[name](http://www.some.com/)'

    finally:
        remove(w.name)


def test_h1_to_h6(tmpdir):
    try:

        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8',
                                         delete=False) as w:
            md = MarkdownWriter(w)
            md.h1('h1')
            md.h2('h2')
            md.h3('h3')
            md.h4('h4')
            md.h5('h5')
            md.h6('h6')

        with open(w.name, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert '# h1\n\n' in content
            assert '# h2\n\n' in content
            assert '# h3\n\n' in content
            assert '# h4\n\n' in content
            assert '# h5\n\n' in content
            assert '# h6\n\n' in content

    finally:
        remove(w.name)


def test_headline(tmpdir):
    with pytest.raises(ValueError):
        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8') as w:
            md = MarkdownWriter(w)
            md.headline(-1, 'headline')


def test_implicit_front_matter(tmpdir):
    try:

        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8',
                                         delete=False) as w:
            md = MarkdownWriter(w, jekyll=True, kramdown=True)
            text = 'something'
            md.write(text)
        with open(w.name, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert content.startswith('---\n')
            assert '---\n\n' in content
            assert text in content

    finally:
        remove(w.name)


def test_strong(tmpdir):
    try:

        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8',
                                         delete=False) as w:
            md = MarkdownWriter(w, jekyll=False, kramdown=False)
            text = 'something __'
            md.strong(text)
        with open(w.name, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert content == '__{}__'.format(text.replace('_', '\\_'))

    finally:
        remove(w.name)


def test_emphasize(tmpdir):
    try:

        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8',
                                         delete=False) as w:
            md = MarkdownWriter(w, jekyll=False, kramdown=False)
            text = 'something **'
            md.emphasize(text)
        with open(w.name, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert content == '*{}*'.format(text.replace('*', '\\*'))

    finally:
        remove(w.name)


def test_blockquote(tmpdir):
    try:

        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8',
                                         delete=False) as w:
            md = MarkdownWriter(w, jekyll=False, kramdown=False)
            text = 'something'
            md.blockquote(text)
        with open(w.name, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert content == '> {}\n\n'.format(text)

    finally:
        remove(w.name)


def test_to_link(tmpdir):
    with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                     mode='w',
                                     encoding='utf-8') as w:
        md = MarkdownWriter(w, jekyll=False, kramdown=False)
        name = 'name'
        target = 'target'
        link = md.to_link(name, target)
        assert link == '[{}]({})'.format(name, target)


def test_link(tmpdir):
    try:

        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8',
                                         delete=False) as w:
            md = MarkdownWriter(w, jekyll=False, kramdown=False)
            name = 'name'
            target = 'target'
            md.link(name, target)
        with open(w.name, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert content == '[{}]({})'.format(name, target)

    finally:
        remove(w.name)


def test_headline_link_github(tmpdir):
    try:

        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8',
                                         delete=False) as w:
            md = MarkdownWriter(w, jekyll=False, kramdown=False)
            headline = 'Headline-1_1 A'
            md.headline_link(headline)
        with open(w.name, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert content == '[{}]({})'.format(headline, '#' + headline.lower()
                                                .replace(' ', '-'))

    finally:
        remove(w.name)


def test_headline_link_kramdown(tmpdir):
    try:

        with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                         mode='w',
                                         encoding='utf-8',
                                         delete=False) as w:
            md = MarkdownWriter(w, jekyll=False, kramdown=True)
            headline = 'Headline-1.1/A'
            md.headline_link(headline)
        with open(w.name, mode='r', encoding='utf-8') as r:
            content = r.read()
            assert content == '[{}]({})'.format(headline, '#' + headline
                                                .lower()
                                                .replace(' ', '-')
                                                .replace('.', '')
                                                .replace('/', ''))

    finally:
        remove(w.name)


def test_figure_no_image(tmpdir):
    try:

        with pytest.raises(ValueError):
            with tempfile.NamedTemporaryFile(dir=str(tmpdir),
                                             mode='w',
                                             encoding='utf-8',
                                             delete=False) as w:
                md = MarkdownWriter(w, jekyll=False, kramdown=False)
                md.figure()

    finally:
        remove(w.name)
