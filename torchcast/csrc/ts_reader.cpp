#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.h"


namespace py = pybind11;


std::string_view inline parse_label(std::string_view& buff_view,
                                    std::vector<long int>& class_labels,
                                    const std::vector<std::string>& class_names)
{
    // Parses a buffer to extract the class label, trimming the buffer to leave
    // behind the remainder. If the class label is not found or is not valid,
    // raises a Python ValueError. The string is expected to be of the form:
    //
    // series_1:series_2:...:class label
    //
    // Args:
    //     buff_view (const std::string_view&): The string to look up.
    //     class_labels (std::vector<int>&): The list of labels to add the new
    //     label to.
    //     class_names (std::vector<std::string>&): The list of strings to look
    //     it up in.

    // Find the colon
    size_t idx { buff_view.rfind(':') };
    if (idx == std::string::npos)
        raise_py_error(
            PyExc_ValueError,
            "Parse error: " + static_cast<std::string>(buff_view)
        );

    // Split on colon into the label and the remainder. Note that, since
    // buff_view is a reference, this trims the buff_view in the calling
    // function.
    std::string_view label { buff_view.substr(idx + 1) };
    buff_view = buff_view.substr(0, idx);

    // Look up the label in the roster and return if found.
    for(size_t i = 0; i < class_names.size(); i++)
        if (label == class_names[i])
        {
            class_labels.push_back(static_cast<long int>(i));
            return buff_view;
        }

    // If not found, error time.
    raise_py_error(
        PyExc_ValueError,
        "Parse error: " + static_cast<std::string>(buff_view)
    );

    return buff_view;
}


py::object parse_ts_body_sparse(std::istream& reader, const bool has_classes,
                                const std::vector<std::string>& class_names,
                                std::vector<long int>& class_labels,
                                const size_t n_t, const size_t n_dim,
                                const bool allow_missing)
{
    if (n_dim == 0)
        raise_py_error(
            PyExc_ValueError,
            "Zero dimensions in time series"
        );
    if (n_t <= 0)
        raise_py_error(
            PyExc_ValueError,
            "Sparse dataset requires time length be specified"
        );
    if (!allow_missing)
        raise_py_error(
            PyExc_ValueError,
            "Sparse dataset requires missing values be allowed"
        );

    // This will be the buffer holding the current line we're reading.
    std::string buff { };
    // This will be a view on the buffer. We're never actually modifying the
    // buffer, but we want to trim it as we parse different pieces, so it
    // should be more efficient to do that on a string_view to prevent copying.
    std::string_view buff_view { };

    // To ensure efficient layout of series in memory - and to allow conversion
    // to a 2-dimensional numpy array if equal_length == true - we store the
    // series as a single long vector, adding each series on to the back as we
    // go.
    std::vector<float> series;
    // Then, we store the break points between series here, which we can use if
    // equal_length == false to efficiently break it up to return as a list of
    // 1-dimensional arrays.
    std::vector<ssize_t> series_breaks { 0 };

    // Iterate through file body.
    while (reader)
    {
        // Load next line and strip whitespace and comments.
        std::getline(reader, buff);
        buff_view = strip_whitespace(strip_comments(buff));

        // Skip blank lines.
        if (buff_view.length() == 0)
            continue;

        // Retrieve class label if present. Recall that this comes off the END.
        if (has_classes)
        {
            buff_view = parse_label(buff_view, class_labels, class_names);
            buff_view = strip_whitespace(buff_view);
        }

        // Parse main series body. First, we iterate over the channels.
        for(size_t i = 0; i < n_dim; i++)
        {
            size_t row_begin { series.size() };

            // For each channel, create the entries in series, filled with
            // NaNs.
            for(size_t t = 0; t < n_t; t++)
                series.push_back(NAN);

            // Check if there's anything to do.
            if ((buff_view.length() == 0) || (buff_view[0] == ':'))
                continue;
            if (buff_view[0] != '(')
                raise_py_error(PyExc_ValueError, "Parse error: " + buff);

            // Now go through the entries to fill them in.
            while (buff_view.length() > 0)
            {
                if (buff_view[0] == ':')
                {
                    buff_view = strip_whitespace(buff_view.substr(1));
                    break;
                }

                // Extract index, value pair and add to series.
                size_t next_comma { buff_view.find(',') };
                size_t next_paran { buff_view.find(')') };
                size_t idx { static_cast<size_t>(extract_int(
                    strip_whitespace(buff_view.substr(1, next_comma - 1))
                )) };
                float value = extract_float(strip_whitespace(
                    buff_view.substr(next_comma + 1,
                                     next_paran - (next_comma + 1))
                ));
                // Should we check if this has already been filled?
                series[row_begin + idx] = value;

                // Drop that pair.
                buff_view = strip_whitespace(buff_view.substr(next_paran + 1));

                // Drop comma.
                if (buff_view[0] == ',')
                    buff_view = strip_whitespace(buff_view.substr(1));
            }
        }

        series_breaks.push_back( static_cast<ssize_t>(series.size()) );
    }

    return coerce_vector_to_py_object(
        series, series_breaks, true, n_dim
    );
}


py::object parse_ts_body_dense(std::istream& reader, const bool has_classes,
                               const std::vector<std::string>& class_names,
                               std::vector<long int>& class_labels,
                               const size_t n_t, const size_t n_dim,
                               const bool allow_missing,
                               const bool equal_length)
{
    if (n_dim == 0)
        raise_py_error(
            PyExc_ValueError,
            "Zero dimensions in time series"
        );

    // This will be the buffer holding the current line we're reading.
    std::string buff { };
    // This will be a view on the buffer. We're never actually modifying the
    // buffer, but we want to trim it as we parse different pieces, so it
    // should be more efficient to do that on a string_view to prevent copying.
    std::string_view buff_view { };

    // To ensure efficient layout of series in memory - and to allow conversion
    // to a 2-dimensional numpy array if equal_length == true - we store the
    // series as a single long vector, adding each series on to the back as we
    // go.
    std::vector<float> series;
    // Then, we store the break points between series here, which we can use if
    // equal_length == false to efficiently break it up to return as a list of
    // 1-dimensional arrays.
    std::vector<ssize_t> series_breaks { 0 };

    // Iterate through file body.
    while (reader)
    {
        // Load next line and strip whitespace and comments.
        std::getline(reader, buff);
        buff_view = strip_whitespace(strip_comments(buff));

        // Skip blank lines.
        if (buff_view.length() == 0)
            continue;

        // Retrieve class label if present.
        if (has_classes)
            buff_view = parse_label(buff_view, class_labels, class_names);

        parse_comma_separated_line(
            buff_view, series, series_breaks, allow_missing, equal_length, n_t,
            n_dim
        );
    }

    return coerce_vector_to_py_object(
        series, series_breaks, equal_length, n_dim
    );
}


// Header options allowed in a ts file.

const HeaderOptions TS_HEADER_OPTIONS {
    { "@problemName", HeaderType::string },
    { "@timeStamps", HeaderType::boolean },
    { "@missing", HeaderType::boolean },
    { "@univariate", HeaderType::boolean },
    { "@dimensions", HeaderType::integer },
    { "@equalLength", HeaderType::boolean },
    { "@seriesLength", HeaderType::integer },
    { "@classLabel", HeaderType::class_labels }
};


py::tuple parse_ts_stream(std::istream& reader)
{
    // Parses a .ts file. This format is documented at:
    //
    // https://github.com/alan-turing-institute/sktime/blob/main/examples/loading_data.ipynb
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
    // Dictionary that will hold attributes etc. to be returned.
    py::dict rtn_dict {};
    // Holds class names
    std::vector<std::string> class_names {};
    // Whether the data is sparse or not
    bool is_sparse { false };
    // Number of time steps
    size_t n_t { 0 };
    // Number of dimensions
    size_t n_dim { 0 };
    // Whether it uses class labels
    bool has_classes { false };
    // If it uses class labels, this will keep track of what label each row is.
    std::vector<long int> class_labels {};
    // Whether each row must be equal length.
    bool equal_length { false };
    // Whether missing variables are allowed.
    bool allow_missing { true };

    // Parse metadata header
    parse_header(
        reader,
        rtn_dict,
        TS_HEADER_OPTIONS,
        nullptr,
        nullptr,
        &class_names
    );

    // Parse metadata to prep for parsing main body
    if (rtn_dict.contains("timeStamps"))
        is_sparse = rtn_dict["timeStamps"].cast<bool>();
    if (rtn_dict.contains("seriesLength"))
        n_t = static_cast<size_t>(rtn_dict["seriesLength"].cast<int>());
    if (rtn_dict.contains("dimensions"))
        n_dim = static_cast<size_t>(rtn_dict["dimensions"].cast<int>());
    if (rtn_dict.contains("univariate"))
        if (rtn_dict["univariate"].cast<bool>())
        {
            if ((n_dim != 0) && (n_dim != 1))
                raise_py_error(
                    PyExc_ValueError,
                    "Conflicting dimension information"
                );
            n_dim = 1;
        }
    if (rtn_dict.contains("classLabel"))
    {
        has_classes = rtn_dict["classLabel"].cast<bool>();
        if (has_classes)
            rtn_dict["classLabel"] = class_names;
    }
    if (rtn_dict.contains("missing"))
        allow_missing = rtn_dict["missing"].cast<bool>();
    if (rtn_dict.contains("equalLength"))
        equal_length = rtn_dict["equalLength"].cast<bool>();

    // If number of dimensions is not specified, peek ahead to find out.
    if (n_dim == 0)
    {
        n_dim = count_columns(reader);
        if (has_classes)
            n_dim--;
    }

    py::object rtn_series;

    if (is_sparse)
        rtn_series = parse_ts_body_sparse(
            reader, has_classes, class_names, class_labels, n_t, n_dim,
            allow_missing
        );
    else
        rtn_series = parse_ts_body_dense(
            reader, has_classes, class_names, class_labels, n_t, n_dim,
            allow_missing, equal_length
        );

    // Add class labels to rtn_dict
    if (has_classes)
    {
        std::vector<ssize_t> shape {
            static_cast<ssize_t>(class_labels.size())
        };
        py::dtype dtype { "int64" };
        py::array rtn_labels { dtype, shape, class_labels.data() };
        rtn_dict["labels"] = rtn_labels;
    }

    return py::make_tuple(rtn_series, rtn_dict);
}


py::tuple load_ts_file(const std::string& path)
{
    // Check if file exists.
    if (!std::filesystem::exists(path))
        raise_py_error(PyExc_FileNotFoundError, "File not found");

    // Create file reader.
    std::ifstream reader { path };
    if (!reader)
        raise_py_error(PyExc_RuntimeError, "Could not open file");

    return parse_ts_stream(reader);
}


py::tuple parse_ts(const std::string& contents)
{
    std::stringstream os { contents };
    return parse_ts_stream(os);
}
