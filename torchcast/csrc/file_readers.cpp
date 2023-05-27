// This file provides support for fast parsing of tsf-formatted files. For
// details of the tsf format, see:
//
//     https://github.com/rakshitha123/TSForecasting/

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using chrono = std::chrono::system_clock;


enum class AttrType
{
    // Used to encode the type of an attribute.

    string,
    numeric,
    date,
};


std::string_view strip_whitespace(const std::string_view& buff)
{
    // This function strips whitespace from either end of a string_view.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     strip whitespace from.
    //
    // Returns:
    //     The :class:`string_view` without leading or trailing whitespace.

    static size_t idx;
    idx = buff.find_first_not_of("\n\r\t ");
    if (idx == std::string::npos)
        return buff.substr(0, 0);
    else
        return buff.substr(idx, buff.find_last_not_of("\n\r\t ") + 1 - idx);
}


std::string_view strip_comments(const std::string_view& buff)
{
    // This function strips comments from the end of a string_view.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     strip comments from.
    //
    // Returns:
    //     The :class:`string_view` without trailing comments.

    static size_t idx {};
    idx = buff.find('#');
    if (idx != std::string::npos)
        return buff.substr(0, idx);
    else
        return buff;
}


void raise_py_error(PyObject* err_type, const std::string& message)
{
    // This is a convenience function for raising a Python layer error, since
    // the built-ins for PyBind11 make it difficult to specify an error
    // message.
    //
    // Args:
    //     err_type (:class:`PyObject`*): The type of Python error to raise.
    //     For a list of options, see:
    //
    //         https://docs.python.org/3/c-api/exceptions.html
    //
    //     message (const :class:`std::string_view`&): The message to include
    //     in the error.

    PyErr_SetString(err_type, message.c_str());
    throw py::error_already_set();
}


bool extract_bool(const std::string_view& buff)
{
    // Converts a string to a boolean value, raising a Python ValueError if it
    // is unable to.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     extract the boolean from.

    if (buff == "true")
        return true;
    else if (buff == "false")
        return false;

    raise_py_error(
        PyExc_ValueError,
        "Parse error: " + static_cast<std::string>(buff)
    );

    // Dummy return value
    return false;
}


chrono::time_point extract_date(const std::string_view& buff)
{
    // Converts a string to a :class:`std::chrono::system_clock::time_point`,
    // which can be automatically converted to a :class:`datetime.datetime` in
    // Python by PyBind11.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     extract the datetime from.

    // TODO: This is inefficient, and induces a copy of buff...
    std::istringstream ss { static_cast<std::string>(buff) };
    static std::tm tm { };
    ss >> std::get_time(&tm, "%Y-%m-%d %H-%M-%S");

    if (ss.fail() || !ss.eof())
        raise_py_error(
            PyExc_ValueError,
            "Parse error: " + static_cast<std::string>(buff)
        );

    return chrono::from_time_t(std::mktime(&tm));
}


int64_t extract_int(const std::string_view& buff)
{
    // Converts a string to an integer value, raising a Python ValueError if it
    // is unable to.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     extract the integer from.

    static std::from_chars_result result;
    static int64_t out;

    result = std::from_chars(buff.begin(), buff.end(), out);

    if ((result.ptr != buff.end()) || (result.ec != std::errc()))
        raise_py_error(
            PyExc_ValueError,
            "Parse error: " + static_cast<std::string>(buff)
        );

    return out;
}


float extract_float(const std::string_view& buff, const bool& allow_missing)
{
    // Converts a string to a float value, raising a Python ValueError if it
    // is unable to.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     extract the integer from.
    //     allow_missing (const bool&): Whether to allow missing values.

    static char* end;
    static float out;

    if (buff == "?")
    {
        if (!allow_missing)
            raise_py_error(
                PyExc_ValueError,
                "Parse error: missing value when missing not allowed"
            );

        return NAN;
    }
    else
    {
        out = std::strtof(buff.begin(), &end);

        if (end != buff.end())
            raise_py_error(
                PyExc_ValueError,
                "Parse error: " + static_cast<std::string>(buff)
            );

        return out;
    }
}


size_t count_columns(const std::string_view& buff)
{
    // Counts the number of columns in a :class:`std::string_view``, by
    // counting the number of commas.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     count the columns in.
    size_t num_cols { 1 };
    for(size_t idx = 0; idx < buff.length(); idx++)
        if (buff[idx] == ',')
            num_cols++;
    return num_cols;
}


std::tuple<int64_t, bool, bool, std::vector<AttrType>,
           std::vector<std::string>>
    parse_tsf_header(std::istream& reader)
{
    // Iterate through file header. Valid keys in the tsf header:
    //
    // attribute (str, str). Valid options:
    //   "string"
    //   "numeric"
    //   "date" (NOT YET SUPPORTED)
    // frequency (str): This is not acutally used.
    // horizon (int): Forecast horizon.
    // missing (bool): This indicates whether some values have been replaced by
    // '?', indicating a NaN.
    // equallength (bool): Whether all series have equal length. This allows
    // some optimization if true.
    // relation (str): This is NOT in the spec, but is found in the San
    // Francisco traffic dataset.
    //
    // The header is generally short, so we don't need to worry about
    // efficiency here.

    // This will be the buffer holding the current line we're reading.
    std::string buff {};
    // This will be a view on the buffer. We're never actually modifying the
    // buffer, but we want to trim it as we parse different pieces, so it
    // should be more efficient to do that on a string_view to prevent copying.
    std::string_view buff_view {};
    // Forecast horizon.
    int64_t horizon { -1 };
    // Whether to allow missing values.
    bool allow_missing { true };
    // Whether all series have equal length.
    bool equal_length { false };
    // Metadata for tsf file.
    std::vector<AttrType> attr_types {};
    std::vector<std::string> attr_names {};
    // Tracks whether we've seen these metadata entries yet.
    bool seen_frequency { false };
    bool seen_horizon { false };
    bool seen_missing { false };
    bool seen_equal_length { false };

    while (reader)
    {
        std::getline(reader, buff);
        buff_view = buff;

        // We begin by stripping out comments and whitespace.
        buff_view = strip_whitespace(strip_comments(buff_view));

        // Valid keys.
        if (buff_view.length() == 0)
            continue;
        else if (buff_view.starts_with("@relation "))
            continue;
        else if (buff_view.starts_with("@frequency "))
        {
            if (seen_frequency)
                raise_py_error(PyExc_ValueError, "Duplicate @frequency key");
            else
                seen_frequency = true;
        }
        else if (buff_view.starts_with("@horizon "))
        {
            if (seen_horizon)
                raise_py_error(PyExc_ValueError, "Duplicate @horizon key");
            else
                seen_horizon = true;
            horizon = extract_int(buff_view.substr(9));
        }
        else if (buff_view.starts_with("@missing "))
        {
            if (seen_missing)
                raise_py_error(PyExc_ValueError, "Duplicate @missing key");
            else
                seen_missing = true;
            allow_missing = extract_bool(buff_view.substr(9));
        }
        else if (buff_view.starts_with("@equallength "))
        {
            if (seen_equal_length)
                raise_py_error(PyExc_ValueError, "Duplicate @equallength key");
            else
                seen_equal_length = true;
            equal_length = extract_bool(buff_view.substr(13));
        }
        else if (buff_view.starts_with("@attribute "))
        {
            // Strip out "@attribute " and any additional whitespace.
            buff_view = buff_view.substr(11);
            buff_view = strip_whitespace(buff_view);
            // Split into two keys.
            size_t idx = buff_view.find(' ');
            if (idx == std::string::npos)
                raise_py_error(PyExc_ValueError, "Parse error: " + buff);
            // Add name to attributes.
            attr_names.push_back(
                static_cast<std::string>(buff_view.substr(0, idx))
            );
            // Add type.
            buff_view = buff_view.substr(buff_view.find_last_of(" \t") + 1);
            if (buff_view == "string")
                attr_types.push_back(AttrType::string);
            else if (buff_view == "numeric")
                attr_types.push_back(AttrType::numeric);
            else if (buff_view == "date")
                attr_types.push_back(AttrType::date);
            else
                raise_py_error(PyExc_ValueError, "Parse error: " + buff);
        }
        else if (buff_view == "@data")
            break;
        else
            raise_py_error(PyExc_ValueError, "Parse error:" + buff);
    }

    return std::make_tuple(
        horizon, allow_missing, equal_length, attr_types, attr_names
    );
}


py::tuple parse_tsf_stream(std::istream& reader)
{
    // Parses a .tsf file, based on the .ts file format. This format is
    // documented at:
    //
    //    https://github.com/rakshitha123/TSForecasting
    //
    // Args:
    //     reader (:class:`std::istream`&): A stream holding the contents of
    //     the file.

    // This will be the buffer holding the current line we're reading.
    std::string buff {};
    // This will be a view on the buffer. We're never actually modifying the
    // buffer, but we want to trim it as we parse different pieces, so it
    // should be more efficient to do that on a string_view to prevent copying.
    std::string_view buff_view {};
    // Buffer for indexing into buff.
    size_t idx;
    // Buffers used to track number of rows and columns in the data section of
    // the tsf.
    size_t num_rows { 0 };
    size_t num_cols { 0 };

    // horizon: int64_t
    // allow_missing: bool
    // equal_length: bool
    // attr_types: std::vector<AttrType>
    // attr_names: std::vector<std::string>
    const auto [horizon, allow_missing, equal_length, attr_types, attr_names] =
        parse_tsf_header(reader);

    // Metadata checking
    if (attr_names.size() == 0)
        raise_py_error(PyExc_ValueError, "Missing attributes section");

    // Create return buffers
    // TODO: Estimate number of entries to pre-allocate space
    std::vector<std::vector<int64_t>> attrs_numeric;
    std::vector<std::vector<std::string>> attrs_string;
    std::vector<std::vector<chrono::time_point>> attrs_date;
    // To ensure efficient layout of series in memory - and to allow conversion
    // to a 2-dimensional numpy array if equal_length == true - we store the
    // series as a single long vector, adding each series on to the back as we
    // go.
    std::vector<float> series;
    // Then, we store the break points between series here, which we can use if
    // equal_length == false to efficiently break it up to return as a list of
    // 1-dimensional arrays.
    std::vector<ssize_t> series_breaks { 0 };
    size_t i_attrs_numeric, i_attrs_string, i_attrs_date;

    for(size_t i_column = 0; i_column < attr_types.size(); i_column++)
    {
        if (attr_types[i_column] == AttrType::string)
            attrs_string.push_back(*(new std::vector<std::string> {}));
        else if (attr_types[i_column] == AttrType::numeric)
            attrs_numeric.push_back(*(new std::vector<int64_t> {}));
        else if (attr_types[i_column] == AttrType::date)
            attrs_date.push_back(*(new std::vector<chrono::time_point> {}));
    }

    // Iterate through file body.
    while (reader)
    {
        // Load next line and strip whitespace and comments.
        std::getline(reader, buff);
        buff_view = buff;
        buff_view = strip_whitespace(strip_comments(buff_view));

        // Skip blank lines.
        if (buff_view.length() == 0)
            continue;

        // Parse attributes.
        i_attrs_numeric = 0;
        i_attrs_string = 0;
        i_attrs_date = 0;
        for(size_t i_column = 0; i_column < attr_types.size(); i_column++)
        {
            idx = buff_view.find(':');
            if (idx == std::string::npos)
                raise_py_error(PyExc_ValueError, "Parse error: " + buff);
            else if (attr_types[i_column] == AttrType::string)
                attrs_string[i_attrs_string++].push_back(
                    static_cast<std::string>(buff_view.substr(0, idx))
                );
            else if (attr_types[i_column] == AttrType::numeric)
                attrs_numeric[i_attrs_numeric++].push_back(
                    extract_int(buff_view.substr(0, idx))
                );
            else if (attr_types[i_column] == AttrType::date)
                attrs_date[i_attrs_date++].push_back(
                    extract_date(buff_view.substr(0, idx))
                );
            buff_view = buff_view.substr(idx + 1);
        }

        // First, count entries so we can allocate the proper array.
        if ((!equal_length) || (num_rows == 0))
            num_cols = count_columns(buff_view);
        // Fill array
        for(size_t i_column = 0; i_column < num_cols - 1; i_column++)
        {
            idx = buff_view.find(',');
            if (idx == std::string::npos)
                raise_py_error(PyExc_ValueError, "Parse error: " + buff);
            series.push_back(
                extract_float(buff_view.substr(0, idx), allow_missing)
            );
            buff_view = buff_view.substr(idx + 1);
        }
        series.push_back(extract_float(buff_view, allow_missing));
        series_breaks.push_back(static_cast<ssize_t>(series.size()));

        num_rows++;
    }

    // Assemble return value

    i_attrs_numeric = 0;
    i_attrs_string = 0;
    i_attrs_date = 0;
    py::dict rtn_dict {};
    py::str dict_key { "horizon" };
    if (horizon >= 0)
        rtn_dict[dict_key] = py::int_ { horizon };
    for(size_t i_column = 0; i_column < attr_types.size(); i_column++)
    {
        dict_key = attr_names[i_column];
        if (attr_types[i_column] == AttrType::string)
            // This triggers a copy, but there's no alternative I'm aware of.
            // It will be returned as a List[str].
            rtn_dict[dict_key] = py::cast(
                attrs_string[i_attrs_string++]
            );
        else if (attr_types[i_column] == AttrType::numeric)
            // I *believe* this does not trigger a copy.
            // TODO: Does this leak memory?
            rtn_dict[dict_key] = py::array_t<int64_t> {
                std::vector<ssize_t> { static_cast<ssize_t>(num_rows) },
                attrs_numeric[i_attrs_numeric++].data()
            };
        else if (attr_types[i_column] == AttrType::date)
            // This triggers a copy, but there's no alternative I'm aware of.
            // It will be returned as a List[datetime.datetime].
            rtn_dict[dict_key] = attrs_date[i_attrs_date++];
    }

    if (equal_length)
    {
        std::vector<ssize_t> shape {
            static_cast<ssize_t>(num_rows),
            static_cast<ssize_t>(num_cols),
        };
        py::array_t<float> rtn_series_array { shape, series.data() };
        return py::make_tuple( rtn_series_array, rtn_dict );
    }
    else
    {
        std::vector<py::array_t<float>> rtn_series_list {};
        ssize_t num_entries;

        for(size_t i_row = 0; i_row < num_rows; i_row++)
        {
            num_entries = series_breaks[i_row + 1] - series_breaks[i_row];
            rtn_series_list.push_back(
                py::array_t<float> {
                    std::vector<ssize_t> { num_entries },
                    series.data() + series_breaks[i_row],
                }
            );
        }

        return py::make_tuple( rtn_series_list, rtn_dict );
    }
}


py::tuple load_tsf_file(const std::string& path)
{
    // Check if file exists.
    if (!std::filesystem::exists(path))
        raise_py_error(PyExc_FileNotFoundError, "File not found");

    // Create file reader.
    std::ifstream reader { path };
    if (!reader)
        raise_py_error(PyExc_RuntimeError, "Could not open file");

    return parse_tsf_stream(reader);
}


py::tuple parse_tsf(const std::string& contents)
{
    std::stringstream os { contents };
    return parse_tsf_stream(os);
}


PYBIND11_MODULE(_C, m) {
    m.def("load_tsf_file", &load_tsf_file, "Reads a tsf file from disk",
          py::arg("path"));
    m.def("parse_tsf", &parse_tsf, "Parses a string formatted as a tsf file",
          py::arg("contents"));
}
