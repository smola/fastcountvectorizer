#!/bin/bash
# This script is intended to run in the continuous integration workflow.
set -e -x

/opt/python/${PYVER}/bin/pip install --upgrade pip

/opt/python/${PYVER}/bin/pip install pybind11

/opt/python/${PYVER}/bin/pip wheel /repo -w wheelhouse/

for whl in wheelhouse/fastcountvectorizer*.whl; do
    auditwheel repair "$whl" --plat ${PLAT} -w /repo/wheelhouse/
done

