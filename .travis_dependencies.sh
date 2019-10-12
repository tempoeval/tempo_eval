#!/bin/bash

ENV_NAME="test-environment"
set -e

conda_create ()
{
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda info -a
    conda env create -q --name $ENV_NAME --file $1
    conda update --all
}

src="$HOME/env/miniconda$TRAVIS_PYTHON_VERSION"
if [ ! -d "$src" ]; then
    sed -i "s/python=3.7/python=$TRAVIS_PYTHON_VERSION/" environment.yml
    ENV_FILE="`pwd`/environment.yml"
    mkdir -p $HOME/env
    pushd $HOME/env

        # Download miniconda packages
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;

        # Install both environments
        bash miniconda.sh -b -p $src

        export PATH="$src/bin:$PATH"
        conda_create $ENV_FILE

        source activate $ENV_NAME
        source deactivate
    popd
else
    echo "Using cached dependencies"
fi