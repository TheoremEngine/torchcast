#pragma once
#include <string>
#include <sstream>
#include <vector>
#include <pybind11/pybind11.h>

#include "utils.h"


namespace py = pybind11;


py::tuple parse_ts_stream(std::istream& reader);
py::tuple load_ts_file(const std::string& path);
py::tuple parse_ts(const std::string& contents);
