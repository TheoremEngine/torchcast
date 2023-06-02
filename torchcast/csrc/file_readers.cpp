#include "ts_reader.h"
#include "tsf_reader.h"
#include "pybind11/pybind11.h"


PYBIND11_MODULE(_file_readers, m) {
    m.def("load_ts_file", &load_ts_file, "Reads a ts file from disk",
          py::arg("path"));
    m.def("load_tsf_file", &load_tsf_file, "Reads a tsf file from disk",
          py::arg("path"));
    m.def("parse_ts", &parse_ts, "Parses a string formatted as a ts file",
          py::arg("contents"));
    m.def("parse_tsf", &parse_tsf, "Parses a string formatted as a tsf file",
          py::arg("contents"));
}
