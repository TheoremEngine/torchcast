#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ts_reader.h"
#include "utils.h"

namespace py = pybind11;


// Header options allowed in a tsf file.

const HeaderOptions TSF_HEADER_OPTIONS {
    { "@attribute", HeaderType::attribute_pair },
    { "@frequency", HeaderType::string },
    { "@horizon", HeaderType::integer },
    { "@missing", HeaderType::boolean },
    { "@equallength", HeaderType::boolean },
    // This is NOT in the spec, but is found in the San Francisco
    // traffic dataset.
    { "@relation", HeaderType::string }
};


py::object parse_tsf_body(std::istream& reader, py::dict& rtn_dict,
                          const std::vector<AttrType>& attr_types,
                          const std::vector<std::string>& attr_names,
                          const bool& allow_missing, const bool& equal_length)
{
    // This will be the buffer holding the current line we're reading.
    std::string buff { };
    // This will be a view on the buffer. We're never actually modifying the
    // buffer, but we want to trim it as we parse different pieces, so it
    // should be more efficient to do that on a string_view to prevent copying.
    std::string_view buff_view { };

    // Create return buffers for attributes.
    std::vector<std::vector<int64_t>> attrs_numeric { };
    std::vector<std::vector<std::string>> attrs_string { };
    std::vector<std::vector<chrono::time_point>> attrs_date { };
    // Indices for indexing attribute buffers.
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

    // To ensure efficient layout of series in memory - and to allow conversion
    // to a 2-dimensional numpy array if equal_length == true - we store the
    // series as a single long vector, adding each series on to the back as we
    // go.
    std::vector<float> series;
    // Then, we store the break points between series here, which we can use if
    // equal_length == false to efficiently break it up to return as a list of
    // 1-dimensional arrays.
    std::vector<ssize_t> series_breaks { 0 };

    while (reader)
    {
        // Load next line and strip whitespace and comments.
        std::getline(reader, buff);
        buff_view = strip_whitespace(strip_comments(buff));

        // Skip blank lines.
        if (buff_view.length() == 0)
            continue;

        // Parse attributes.
        i_attrs_numeric = 0;
        i_attrs_string = 0;
        i_attrs_date = 0;
        for(size_t i_column = 0; i_column < attr_types.size(); i_column++)
        {
            size_t idx = buff_view.find(':');
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

        parse_comma_separated_line(
            buff_view, series, series_breaks, allow_missing, equal_length, 0, 1
        );
    }

    // Assemble return value

    i_attrs_numeric = 0;
    i_attrs_string = 0;
    i_attrs_date = 0;
    ssize_t num_rows { static_cast<ssize_t>(series_breaks.size()) - 1 };
    for(size_t i_column = 0; i_column < attr_types.size(); i_column++)
    {
        py::str dict_key { attr_names[i_column] };
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
                std::vector<ssize_t> { num_rows },
                attrs_numeric[i_attrs_numeric++].data()
            };
        else if (attr_types[i_column] == AttrType::date)
            // This triggers a copy, but there's no alternative I'm aware of.
            // It will be returned as a List[datetime.datetime].
            rtn_dict[dict_key] = attrs_date[i_attrs_date++];
    }

    py::object rtn_series {
        coerce_vector_to_py_object(series, series_breaks, equal_length, 1)
    };
    return rtn_series;
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

    // Dictionary that will hold attributes etc. to be returned.
    py::dict rtn_dict {};
    // Vectors holding types and names of attributes
    std::vector<std::string> attr_names {};
    std::vector<AttrType> attr_types {};
    bool equal_length { false };
    bool allow_missing { true };

    // Parse metadata header
    parse_header(
        reader,
        rtn_dict,
        TSF_HEADER_OPTIONS,
        &attr_names,
        &attr_types,
        nullptr
    );

    if (rtn_dict.contains("equallength"))
        equal_length = rtn_dict["equallength"].cast<bool>();
    if (rtn_dict.contains("allowmissing"))
        allow_missing = rtn_dict["allowmissing"].cast<bool>();

    // Metadata checking
    if (attr_names.size() == 0)
        raise_py_error(PyExc_ValueError, "Missing attributes section");

    py::object rtn_series {
        parse_tsf_body(reader, rtn_dict, attr_types, attr_names, allow_missing,
                       equal_length)
    };

    return py::make_tuple( rtn_series, rtn_dict );
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
