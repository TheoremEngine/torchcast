import os
from setuptools import find_packages, setup

from pybind11.setup_helpers import Pybind11Extension

COMPILE_ARGS = [
    '-Wall',
    '-Wextra',
    '-Wsign-conversion',
]


if __name__ == '__main__':
    # Hypothetically, somebody might run this from somewhere other than the
    # root directory.
    root_path = os.path.dirname(os.path.abspath(__file__))

    # Read in requirements.txt
    req_path = os.path.join(root_path, 'requirements.txt')
    with open(req_path, 'r') as reqs_file:
        requirements = list(reqs_file.readlines())

    # Create pybind11 extension
    ext_modules = [
        Pybind11Extension(
            'torchcast.datasets._file_readers',
            ['torchcast/csrc/file_readers.cpp',
             'torchcast/csrc/utils.cpp',
             'torchcast/csrc/ts_reader.cpp',
             'torchcast/csrc/tsf_reader.cpp'],
            cxx_std=17,
            extra_compile_args=COMPILE_ARGS,
        ),
    ]

    setup(
        name='torchcast',
        version='0.1',
        author='Mark Lowell',
        author_email='MarkLowell@theorem-engine.org',
        packages=find_packages(),
        package_data={'torchcast': ['datasets/*.txt']},
        install_requires=requirements,
        ext_modules=ext_modules,
    )
