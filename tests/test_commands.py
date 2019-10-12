from os import remove
from os.path import join, exists

import jams
import pytest


def test_tempo_eval_help(script_runner):

    # sanity check -- can it be called at all
    ret = script_runner.run('tempo_eval', '--help')
    assert ret.success
    assert ret.stdout.startswith('usage:')
    assert ret.stderr == ''


def test_tempo_eval_gtzan(script_runner, tmpdir):
    output_dir = str(tmpdir)

    ret = script_runner.run('tempo_eval',
                            '-c', 'gtzan',
                            '-d', output_dir,
                            '-s', 'L')
    assert ret.success
    assert exists(join(output_dir, 'index.md'))
    assert exists(join(output_dir, 'gtzan.md'))


def test_convert2jams_help(script_runner):

    # sanity check -- can it be called at all
    ret = script_runner.run('convert2jams', '--help')
    assert ret.success
    assert ret.stdout.startswith('usage:')
    assert ret.stderr == ''


def test_convert2jams_required_args(script_runner, tmpdir, caplog):

    ret = script_runner.run('convert2jams',
                            '-i', str(tmpdir),
                            '-a', 'some_dir')
    assert not ret.success
    assert 2 == ret.returncode  # because corpus is missing
    assert 'No corpus name provided' in caplog.text

    ret = script_runner.run('convert2jams',
                            '-i', str(tmpdir),
                            '-a', 'some_dir',
                            '-c', 'corpus')
    assert not ret.success
    assert 2 == ret.returncode  # because version is missing
    assert 'No annotation version provided' in caplog.text

    ret = script_runner.run('convert2jams',
                            '-i', str(tmpdir),
                            '-a', 'some_dir',
                            '-c', 'corpus',
                            '-v', 'version')
    assert ret.success  # because input dir is empty


def test_convert2jams_unsupported_format(script_runner, tmpdir, caplog):

    csv_file = join(str(tmpdir), 'test.xls')
    with open(csv_file, mode='w', encoding='utf-8') as f:
        f.write('abc,100.5\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-d', str(tmpdir),
                            '-i', csv_file,
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version)

    # unsupported format
    assert not ret.success
    assert 'Input data format is not supported.' in caplog.text


def test_convert2jams_fake_audio_files(script_runner, tmpdir, caplog):

    csv_file = join(str(tmpdir), 'test.csv')
    jams_file = join(str(tmpdir), 'abc.jams')
    with open(csv_file, mode='w', encoding='utf-8') as f:
        f.write('abc,100.5\n')

    with open(join(str(tmpdir), 'abc.wav'), mode='w', encoding='utf-8') as f:
        f.write('FAKE WAVE\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-d', str(tmpdir),
                            '-i', csv_file,
                            '-a', str(tmpdir),
                            '-c', corpus,
                            '-v', version,
                            '--verbose')
    assert ret.success
    jams.load(jams_file)
    assert 'Failed to extract audio duration' in caplog.text


def test_convert2jams_csv(script_runner, tmpdir):

    csv_file = join(str(tmpdir), 'test.csv')
    jams_file = join(str(tmpdir), 'abc.jams')
    with open(csv_file, mode='w', encoding='utf-8') as f:
        f.write('abc,100.5\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-d', str(tmpdir),
                            '-i', csv_file,
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version,
                            '--flatten')
    assert ret.success
    jam = jams.load(jams_file)
    assert jam.annotations[0].annotation_metadata.corpus == corpus
    assert jam.annotations[0].annotation_metadata.version == version
    assert jam.annotations['tempo'][0]['data'][0].value == 100.5


def test_convert2jams_tsv(script_runner, tmpdir):

    tsv_file = join(str(tmpdir), 'test.tsv')
    jams_file = join(str(tmpdir), 'abc.jams')
    with open(tsv_file, mode='w', encoding='utf-8') as f:
        f.write('abc\t100.5\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-i', tsv_file,
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version)
    assert ret.success
    jam = jams.load(jams_file)
    assert jam.annotations[0].annotation_metadata.corpus == corpus
    assert jam.annotations[0].annotation_metadata.version == version
    assert jam.annotations['tempo'][0]['data'][0].value == 100.5


def test_convert2jams_json(script_runner, tmpdir):

    json_file = join(str(tmpdir), 'test.json')
    jams_file0 = join(str(tmpdir), 'abc.jams')
    jams_file1 = join(str(tmpdir), 'def.jams')
    with open(json_file, mode='w', encoding='utf-8') as f:
        f.write('{ "abc": 100.5, "def": [1.0, 10.0, 0.2] }\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-i', json_file,
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version,
                            '--flatten')
    assert ret.success
    jam = jams.load(jams_file0)
    assert jam.annotations[0].annotation_metadata.corpus == corpus
    assert jam.annotations[0].annotation_metadata.version == version
    assert jam.annotations['tempo'][0]['data'][0].value == 100.5

    jam = jams.load(jams_file1)
    assert jam.annotations[0].annotation_metadata.corpus == corpus
    assert jam.annotations[0].annotation_metadata.version == version
    assert jam.annotations['tempo'][0]['data'][0].value == 1.
    assert pytest.approx(0.2, jam.annotations['tempo'][0]['data'][0].confidence )
    assert jam.annotations['tempo'][0]['data'][1].value == 10.
    assert pytest.approx(0.8, jam.annotations['tempo'][0]['data'][1].confidence)


def test_convert2jams_text_files(script_runner, tmpdir):

    text_file0 = join(str(tmpdir), 'abc.txt')
    text_file1 = join(str(tmpdir), 'def.txt')
    jams_file0 = join(str(tmpdir), 'abc.jams')
    jams_file1 = join(str(tmpdir), 'def.jams')
    with open(text_file0, mode='w', encoding='utf-8') as f:
        f.write('50.5\n')
    with open(text_file1, mode='w', encoding='utf-8') as f:
        f.write('50.5 100 0.9\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-i', str(tmpdir),
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version)
    assert ret.success
    jam = jams.load(jams_file0)
    assert jam.annotations[0].annotation_metadata.corpus == corpus
    assert jam.annotations[0].annotation_metadata.version == version
    assert jam.annotations['tempo'][0]['data'][0].value == 50.5

    jam = jams.load(jams_file1)
    assert jam.annotations[0].annotation_metadata.corpus == corpus
    assert jam.annotations[0].annotation_metadata.version == version
    assert jam.annotations['tempo'][0]['data'][0].value == 50.5
    assert pytest.approx(0.9, jam.annotations['tempo'][0]['data'][0].confidence)
    assert jam.annotations['tempo'][0]['data'][1].value == 100.
    assert pytest.approx(0.1, jam.annotations['tempo'][0]['data'][1].confidence)


def test_convert2jams_text_files2(script_runner, tmpdir):

    text_file = join(str(tmpdir), 'abc.txt')
    jams_file = join(str(tmpdir), 'out', 'abc.jams')
    with open(text_file, mode='w', encoding='utf-8') as f:
        f.write('50.5\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            # explicitly set output dir
                            '-d', join(str(tmpdir), 'out'),
                            '-i', str(tmpdir),
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version)
    assert ret.success

    # make sure we can load the results
    jams.load(jams_file)


def test_convert2jams_text_files_flatten(script_runner, tmpdir, caplog):

    text_file = join(str(tmpdir), 'abc.txt')
    with open(text_file, mode='w', encoding='utf-8') as f:
        f.write('50.5\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-i', str(tmpdir),
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version,
                            '--flatten')

    # flatten is not supported for text files (yet)
    assert not ret.success
    assert 'Flatten is currently only supported' in caplog.text


def test_convert2jams_text_files_bad_format(script_runner, tmpdir):

    text_file = join(str(tmpdir), 'abc.txt')
    jams_file = join(str(tmpdir), 'abc.jams')
    with open(text_file, mode='w', encoding='utf-8') as f:
        f.write('50.5 AA 10.\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-i', str(tmpdir),
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version)

    assert ret.success
    # test error log message?
    assert not exists(jams_file)


def test_convert2jams_text_files_with_bpm_extension(script_runner, tmpdir):

    text_file = join(str(tmpdir), 'abc.bpm.txt')  # .bpm.txt extension!
    jams_file = join(str(tmpdir), 'abc.jams')
    with open(text_file, mode='w', encoding='utf-8') as f:
        f.write('50.5 10 .9\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-i', str(tmpdir),
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version)

    assert ret.success
    # make sure it exists and we can load it
    jams.load(jams_file)


def test_convert2jams_template(script_runner, tmpdir, caplog):

    # create template..
    csv_file_template = join(str(tmpdir), 'template.csv')
    with open(csv_file_template, mode='w', encoding='utf-8') as f:
        f.write('abc,100.5\n')
    template = join(str(tmpdir), 'abc.jams')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-i', csv_file_template,
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version,
                            '--verbose')
    assert ret.success

    # use fresh jam as template
    csv_file_test = join(str(tmpdir), 'test.csv')
    with open(csv_file_test, mode='w', encoding='utf-8') as f:
        f.write('def,10.5\n')
    test = join(str(tmpdir), 'def.jams')
    ret = script_runner.run('convert2jams',
                            '-i', csv_file_test,
                            '-a', 'some_dir',
                            '--template', template)
    assert ret.success

    jam = jams.load(test)
    assert jam.annotations[0].annotation_metadata.corpus == corpus
    assert jam.annotations[0].annotation_metadata.version == version
    assert jam.annotations['tempo'][0]['data'][0].value == 10.5

    remove(test)
    with open(template, mode='r', encoding='utf-8') as f:
        template_string = f.read()

    no_corpus_template = join(str(tmpdir), 'no_corpus.jams')
    with open(no_corpus_template, mode='w', encoding='utf-8') as f:
        f.write(template_string.replace('"corpus": "corpus",', ''))

    no_version_template = join(str(tmpdir), 'no_version.jams')
    with open(no_version_template, mode='w', encoding='utf-8') as f:
        f.write(template_string.replace('"version": "version",', ''))

    ret = script_runner.run('convert2jams',
                            '-i', csv_file_test,
                            '-a', 'some_dir',
                            '--template', no_corpus_template)

    assert not ret.success
    assert 'Template annotation metadata must contain corpus name' in caplog.text

    ret = script_runner.run('convert2jams',
                            '-i', csv_file_test,
                            '-a', 'some_dir',
                            '--template', no_version_template)

    assert not ret.success
    assert 'Template annotation metadata must contain version' in caplog.text


def test_convert2jams_bad_template(script_runner, tmpdir, caplog):

    csv_file_template = join(str(tmpdir), 'template.csv')
    with open(csv_file_template, mode='w', encoding='utf-8') as f:
        f.write('abc,100.5\n')

    ret = script_runner.run('convert2jams',
                            '-i', csv_file_template,
                            '-a', 'some_dir',
                            # use csv file as jams template, which cannot work
                            '--template', csv_file_template)

    assert not ret.success
    assert 'Failed to extract annotation metadata from template file' in caplog.text

    ret = script_runner.run('convert2jams',
                            '-i', csv_file_template,
                            '-a', 'some_dir',
                            # use csv file as jams template, which cannot work
                            '--template', 'does_not_exist.jams')

    assert not ret.success
    assert 'Annotation metadata template file does not exists' in caplog.text

    ret = script_runner.run('convert2jams',
                            '-i', csv_file_template,
                            '-a', 'some_dir',
                            # use csv file as jams template, which cannot work
                            '--template', str(tmpdir))

    assert not ret.success
    assert 'Annotation metadata template is not a file' in caplog.text


def test_convert2jams_input_does_not_exist(script_runner, caplog):

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-i', 'some_dir',
                            '-a', 'some_audio_dir',
                            '-c', corpus,
                            '-v', version)
    assert not ret.success
    assert 'Input file/dir does not exists' in caplog.text


def test_convert2jams_bibtex_file(script_runner, tmpdir):

    csv_file = join(str(tmpdir), 'test.csv')
    bib_file = join(str(tmpdir), 'test.bib')
    jams_file = join(str(tmpdir), 'abc.jams')
    with open(csv_file, mode='w', encoding='utf-8') as f:
        f.write('abc,100.5\n')

    bibstuff = 'bibstuff'
    with open(bib_file, mode='w', encoding='utf-8') as f:
        f.write(bibstuff)

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-i', csv_file,
                            '-a', 'some_dir',
                            '-c', corpus,
                            '-v', version,
                            '--bibtex', bib_file)
    assert ret.success
    jam = jams.load(jams_file)
    assert jam.annotations[0].annotation_metadata.annotator['bibtex'] == bibstuff


def test_convert2jams_jam_exists(script_runner, tmpdir, caplog):

    csv_file = join(str(tmpdir), 'test.csv')
    with open(csv_file, mode='w', encoding='utf-8') as f:
        f.write('abc,100.5\n')

    corpus = 'corpus'
    version = 'version'
    ret = script_runner.run('convert2jams',
                            '-i', csv_file,
                            '-a', str(tmpdir),
                            '-c', corpus,
                            '-v', version)
    assert ret.success

    ret = script_runner.run('convert2jams',
                            '-i', csv_file,
                            '-a', str(tmpdir),
                            '-c', corpus,
                            '-v', version)
    assert not ret.success
    assert 'JAMS file already exists' in caplog.text
