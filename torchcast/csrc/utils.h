#pragma once
#include <iomanip>
#include <charconv>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;


enum class AttrType
{
    // Used to encode the type of an attribute in a tsf file.

    string,
    numeric,
    date,
};


enum class HeaderType
{
    // Used to encode the type of a key-value pair in the header.

    // attribute_pair specifies that this line should be passed to the
    // parse_attribute_pair function.
    attribute_pair,
    class_labels,
    boolean,
    integer,
    // skip: Do nothing.
    skip,
    string
};

using chrono = std::chrono::system_clock;

// Header options specifies the allowed key-value pairs in the header. Each
// entry is a name and a HeaderType. All keys are expected to start with "@",
// although this is not enforced. The only key allowed that is not in the
// HeaderOptions is "@data", which indicates the end of the header block and
// start of the data.

using HeaderOptions = std::vector<std::pair<std::string, HeaderType>>;

// ListOfArrays will be coerced to a List[np.ndarray] when returned.

using ListOfArrays = std::vector<py::array_t<float>>;


void inline raise_py_error(PyObject* err_type, const std::string& message)
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


std::string_view inline strip_whitespace(const std::string_view& buff)
{
    // This function strips whitespace from either end of a string_view. This
    // function is expected to always succeed.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     strip whitespace from.

    size_t idx;
    idx = buff.find_first_not_of("\n\r\t ");
    if (idx == std::string::npos)
        return buff.substr(0, 0);
    else
        return buff.substr(idx, buff.find_last_not_of("\n\r\t ") + 1 - idx);
}


std::string_view inline strip_comments(const std::string_view& buff)
{
    // This function strips comments from the end of a string_view. This
    // function is expected to always succeed.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     strip comments from.

    size_t idx {};
    idx = buff.find('#');
    if (idx != std::string::npos)
        return buff.substr(0, idx);
    else
        return buff;
}


bool inline extract_bool(const std::string_view& buff)
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
        "Cannot convert to bool: " + static_cast<std::string>(buff)
    );

    // Dummy return value
    return false;
}


chrono::time_point inline extract_date(const std::string_view& buff)
{
    // Converts a string to a :class:`std::chrono::system_clock::time_point`,
    // which can be automatically converted to a :class:`datetime.datetime` in
    // Python by PyBind11. This raises a Python ValueError if the parse fails.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     extract the datetime from.

    // TODO: This is inefficient, and induces a copy of buff...
    std::istringstream ss { static_cast<std::string>(buff) };
    std::tm tm { };
    ss >> std::get_time(&tm, "%Y-%m-%d %H-%M-%S");

    if (ss.fail() || !ss.eof())
        raise_py_error(
            PyExc_ValueError,
            "Cannot convert to datetime: " + static_cast<std::string>(buff)
        );

    return chrono::from_time_t(std::mktime(&tm));
}


float inline extract_float(const std::string_view& buff,
                           const bool& allow_missing = true)
{
    // Converts a string to a float value, raising a Python ValueError if it
    // is unable to.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     extract the integer from.
    //     allow_missing (const bool&): Whether to allow missing values.

    char* end;
    float out;

    if (buff == "?")
    {
        if (!allow_missing)
            raise_py_error(
                PyExc_ValueError,
                "Missing value when missing not allowed"
            );

        return NAN;
    }
    else
    {
        out = std::strtof(buff.begin(), &end);

        if (end != buff.end())
            raise_py_error(
                PyExc_ValueError,
                "Cannot convert to float: " + static_cast<std::string>(buff)
            );

        return out;
    }
}


int64_t inline extract_int(const std::string_view& buff)
{
    // Converts a string to an integer value, raising a Python ValueError if it
    // is unable to.
    //
    // Args:
    //     buff (const :class:`std::string_view`&): The :class:`string_view` to
    //     extract the integer from.

    std::from_chars_result result;
    int64_t out;

    result = std::from_chars(buff.begin(), buff.end(), out);

    if ((result.ptr != buff.end()) || (result.ec != std::errc()))
        raise_py_error(
            PyExc_ValueError,
            "Cannot convert to int: " + static_cast<std::string>(buff)
        );

    return out;
}


void parse_attribute_pair(std::string_view& buff_view,
                          std::vector<std::string>* attr_names,
                          std::vector<AttrType>* attr_types);
bool parse_class_names(std::string_view& buff_view,
                       std::vector<std::string>* class_names);
void parse_header(std::istream& reader, py::dict& rtn,
                  const HeaderOptions& allowed_keys,
                  std::vector<std::string>* attr_names,
                  std::vector<AttrType>* attr_types,
                  std::vector<std::string>* class_names);
void parse_comma_separated_line(std::string_view& buff_view,
                                std::vector<float>& series,
                                std::vector<ssize_t>& series_breaks,
                                const bool& allow_missing,
                                const bool& equal_length,
                                const size_t& n_t, const size_t& n_dim);
size_t count_columns(std::istream& reader);

py::object coerce_vector_to_py_object(const std::vector<float>& series,
                                      const std::vector<ssize_t>& series_breaks,
                                      const bool& equal_length,
                                      const size_t& n_dim);
