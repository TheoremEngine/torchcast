from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension

COMPILE_ARGS = [
    '-Wall',
    '-Wextra',
    '-Wsign-conversion',
]


if __name__ == '__main__':
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
    setup(ext_modules=ext_modules)
