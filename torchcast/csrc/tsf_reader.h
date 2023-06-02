#pragma once
#include <sstream>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>

#include "utils.h"


namespace py = pybind11;


py::tuple parse_tsf_stream(std::istream& reader);
py::tuple load_tsf_file(const std::string& path);
py::tuple parse_tsf(const std::string& contents);
