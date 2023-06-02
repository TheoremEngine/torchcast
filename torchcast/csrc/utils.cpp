#include <sstream>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>

#include "utils.h"


namespace py = pybind11;


void parse_attribute_pair(std::string_view& buff_view,
                          std::vector<std::string>* attr_names,
                          std::vector<AttrType>* attr_types)
{
    // Parses a line to extract an attribute pair and adds it to the vectors
    // containing them. Attributes are used in tsf file headers and must be of
    // the form:
    //
    // @attribute name type
    //
    // Where type must be chosen from "numeric" (i.e. int), "date", or
    // "string". This function will raise a ValueError if it is unable to parse
    // the line. It will raise a RuntimeError if it is not passed non-null
    // pointers to attr_names and attr_types, which should never happen and
    // indicates a bug.
    //
    // Args:
    //     buff_view (std::string_view&): The view of the line containing the
    //     attribute pair to parse. The "@attribute" key is expected to have
    //     already been stripped out.
    //     attr_names (std::vector<std::string>*): A pointer to a vector
    //     storing the names of the attributes. The new attribute will be added
    //     to this vector.
    //     attr_types (std::vector<AttrType>*): A pointer to a vector storing
    //     the types of the attributes. The new attribute will be added to this
    //     vector.

    if ((attr_names == nullptr) || (attr_types == nullptr))
        raise_py_error(
            PyExc_RuntimeError,
            "attributes in header options but not available"
        );

    // Strip out any additional whitespace.
    buff_view = strip_whitespace(buff_view);

    // Split into key and value.
    size_t idx = buff_view.find(' ');
    if (idx == std::string::npos)
        raise_py_error(
            PyExc_ValueError,
            "Parse error: " + static_cast<std::string>(buff_view)
        );

    // Add name to attr_names.
    (*attr_names).push_back(
        static_cast<std::string>(buff_view.substr(0, idx))
    );

    // Add type to attr_types.
    buff_view = buff_view.substr(buff_view.find_last_of(" \t") + 1);
    if (buff_view == "string")
        (*attr_types).push_back(AttrType::string);
    else if (buff_view == "numeric")
        (*attr_types).push_back(AttrType::numeric);
    else if (buff_view == "date")
        (*attr_types).push_back(AttrType::date);
    else
        raise_py_error(
            PyExc_ValueError,
            "Parse error: " + static_cast<std::string>(buff_view)
        );
}


bool parse_class_names(std::string_view& buff_view,
                       std::vector<std::string>* class_names)
{
    // Parses a line to extract a list of class names. Class names are used in
    // ts file headers and must be one of the following:
    //
    // @classLabel false
    // @classLabel true name1 name2 ...
    //
    // This function will raise a ValueError if it is unable to parse the line.
    // It will raise a RuntimeError if it is not passed a non-null pointer to
    // class_names, which should never happen and indicates a bug.
    //
    // Args:
    //     buff_view (std::string_view&): The view of the line containing the
    //     attribute pair to parse. The "@classLabel" key is expected to have
    //     already been stripped out.
    //     class_names (std::vector<std::string>*): A pointer to a vector
    //     storing the names of the classes. The class names will be added to
    //     this vector if they are present.

    if (class_names == nullptr)
        raise_py_error(
            PyExc_RuntimeError,
            "classLabel in header options but not available"
        );

    if (buff_view == "false")
        return false;

    if (!buff_view.starts_with("true "))
        raise_py_error(
            PyExc_ValueError,
            "Parse error: " + static_cast<std::string>(buff_view)
        );

    // Strip out "true " from buffer
    buff_view = strip_whitespace(buff_view.substr(4));

    if (buff_view.length() == 0)
        raise_py_error(
            PyExc_ValueError,
            "Parse error: class labels true but no class names found"
        );

    size_t idx = buff_view.find(' ');

    while (idx != std::string::npos)
    {
        (*class_names).push_back(
            static_cast<std::string>(buff_view.substr(0, idx))
        );
        buff_view = buff_view.substr(idx);
        buff_view = strip_whitespace(buff_view);
        idx = buff_view.find(' ');
    }
    (*class_names).push_back(static_cast<std::string>(buff_view));

    return true;
}


void parse_header(std::istream& reader, py::dict& rtn,
                  const HeaderOptions& allowed_keys,
                  std::vector<std::string>* attr_names,
                  std::vector<AttrType>* attr_types,
                  std::vector<std::string>* class_names)
{
    // This is a generic parsing function for the headers of tsf and ts files.
    // This function will raise a Python ValueError in the event of a malformed
    // file. It may raise a RuntimeError if allowed_keys contains an
    // attribute_pair key but attr_names or attr_types is a null pointer; this
    // condition should never occur.
    //
    // Args:
    //     reader (std::istream&): The stream holding the file being parsed.
    //     rtn (py::dict&): A Python dictionary holding the extracted key-value
    //     pairs to be returned.
    //     allowed_keys (const HeaderOptions&): Specifies what keys are allowed
    //     in the header. This is specific to the file type.
    //     attr_names (std::vector<std::string>*): A pointer to a vector
    //     storing the names of the attributes. This is only used in tsf files
    //     and should otherwise be a null pointer.
    //     attr_types (std::vector<AttrType>*): A pointer to a vector storing
    //     the types of the attributes. This is only used in tsf files
    //     and should otherwise be a null pointer.
    //     class_names (std::vector<std::string>*) A pointer to a vector
    //     storing the class names. This is only used in ts files and should
    //     otherwise be a null pointer.

    // This will be the buffer holding the current line we're reading.
    std::string buff {};
    // This will be a view on the buffer. We're never actually modifying the
    // buffer, but we want to trim it as we parse different pieces, so it
    // should be more efficient to do that on a string_view to prevent copying.
    std::string_view buff_view {};

    while (reader)
    {
        next_row:

        std::getline(reader, buff);
        buff_view = buff;

        // We begin by stripping out comments and whitespace.
        buff_view = strip_whitespace(strip_comments(buff_view));

        if (buff_view.length() == 0)
            continue;
        // Indicates end of the header block.
        else if (buff_view == "@data")
            break;

        for(size_t i = 0; i < allowed_keys.size(); i++)
        {
            // Iterate through candidate keys until we find the right one.
            std::string_view key { std::get<0>(allowed_keys[i]) };
            if (buff_view.starts_with(key))
            {
                if (buff_view.length() <= key.length() + 1)
                    raise_py_error(
                        PyExc_ValueError,
                        "Key missing value: " + buff
                    );

                // Convert the key to a Python string so it can be used as a
                // key in the Python dictionary. Remember to strip off the @.
                py::str py_key { key.substr(1) };
                // Extract the value type.
                HeaderType value_type { std::get<1>(allowed_keys[i]) };
                // For convenience, crop buff_view down to just the value.
                buff_view = buff_view.substr(key.length() + 1);

                // Check for duplicate keys.
                if (rtn.contains(key))
                    raise_py_error(
                        PyExc_ValueError,
                        "Duplicate key: " + static_cast<std::string>(key)
                    );
                else if (value_type == HeaderType::attribute_pair)
                    parse_attribute_pair(buff_view, attr_names, attr_types);
                else if (value_type == HeaderType::boolean)
                    rtn[py_key] = static_cast<py::bool_>(
                        extract_bool(buff_view)
                    );
                else if (value_type == HeaderType::class_labels)
                    rtn[py_key] = static_cast<py::bool_>(
                        parse_class_names(buff_view, class_names)
                    );
                else if (value_type == HeaderType::integer)
                    rtn[py_key] = static_cast<py::int_>(
                        extract_int(buff_view)
                    );
                else if (value_type == HeaderType::string)
                    rtn[py_key] = static_cast<py::str>(buff_view);
                goto next_row;
            }
        }

        // If we reached the end of the while block, it means we failed to find
        // a match to any of the valid options, indicating a malformed file.
        raise_py_error(PyExc_ValueError, "Parse error: " + buff);
    }
}


void parse_comma_separated_line(std::string_view& buff_view,
                                std::vector<float>& series,
                                std::vector<ssize_t>& series_breaks,
                                const bool& allow_missing,
                                const bool& equal_length,
                                const size_t& n_t = 0,
                                const size_t& n_dim = 1)
{
    // Parses a single buffer containing a series. Each channel in the series
    // is separated by colons, and each entry in each channel is a float
    // separated by commas. The buffer may also contain question marks, which
    // will be interpreted as NaNs. It may contain no other characters, and
    // will raise a Python ValueError if any are encountered.
    //
    // Args:
    //     buff_view (std::string_view&): The buffer to parse.
    //     series (std::vector<float>&): The vector to add extracted values to.
    //     series_breaks (std::vector<ssize_t>&): The vector tracking the ends
    //     of each series.
    //     allow_missing (const bool&): Whether to allow NaN values.
    //     equal_length (const bool&): If true, enforce that each row should
    //     have equal length.
    //     n_t (const ssize_t&): If a non-zero number is provided, the function
    //     will raise a Python ValueError if equal_length is true and any
    //     series does not have this time length.
    //     n_dim (const ssize_t&): The function will raise a Python ValueError
    //     if any series does not have this number of channels.

    size_t comma_idx { };
    size_t colon_idx { };

    for(size_t d = 0; d < n_dim; d++)
    {
        colon_idx = buff_view.find(':');
        if (colon_idx == std::string::npos)
            colon_idx = buff_view.length();
        comma_idx = buff_view.find(',');
        while ((comma_idx != std::string::npos) && (comma_idx < colon_idx))
        {
            series.push_back(
                extract_float(buff_view.substr(0, comma_idx), allow_missing)
            );
            buff_view = buff_view.substr(comma_idx + 1);
            colon_idx -= (comma_idx + 1);
            comma_idx = buff_view.find(',');
        }
        series.push_back(
            extract_float(buff_view.substr(0, colon_idx), allow_missing)
        );
        if (d < n_dim - 1)
            buff_view = buff_view.substr(colon_idx + 1);
    }

    series_breaks.push_back(static_cast<ssize_t>(series.size()));

    if ((equal_length) && (series_breaks.size() > 2))
    {
        size_t n_rows { series_breaks.size() };
        size_t t_row { static_cast<size_t>(
            series_breaks[n_rows - 1] - series_breaks[n_rows - 2]
        ) };
        if (
            (series_breaks[1] - series_breaks[0]) !=
            static_cast<ssize_t>(t_row)
        )
            raise_py_error(PyExc_ValueError, "Incorrect time length");
        if ((n_t != 0) && (t_row != (n_t * n_dim)))
            raise_py_error(PyExc_ValueError, "Incorrect time length");
    }
}


size_t count_columns(std::istream& reader)
{
    // Peeks ahead one line in a stream to count the number of columns.
    //
    // Args:
    //     reader (std::istream&): The reader to use.

    std::string buff { };
    std::string_view buff_view { };
    size_t n_dim { 0 };

    // Record current position.
    ssize_t cur_pos { reader.tellg() };

    // Read ahead until we find a non-empty line.
    while (buff_view.length() == 0)
    {
        std::getline(reader, buff);
        buff_view = strip_whitespace(strip_comments(buff));
    }

    // Count the number of colons.
    while (true)
    {
        n_dim++;
        size_t next_colon { buff_view.find(':') };
        if (next_colon == std::string::npos)
            break;
        buff_view = buff_view.substr(next_colon + 1);
    }

    // Reset back to original position.
    reader.seekg(cur_pos, std::ios_base::beg);

    return n_dim;
}


py::object coerce_vector_to_py_object(const std::vector<float>& series,
                                      const std::vector<ssize_t>& series_breaks,
                                      const bool& equal_length,
                                      const size_t& n_dim)
{
    // This function coerces the vectors created by our parsing of series into
    // Python-acceptable formats. If each row is of equal length, then they are
    // returned as a single :class:`numpy.ndarray`. If they are of unequal
    // length, then they are returned as a list of :class:`numpy.ndarray`.
    //
    // Args:
    //     series (const std::vector<float>&): The single vector holding all of
    //     the floats in the series.
    //     series_breaks (const std::vector<ssize_t>&): The vector tracking the
    //     locations between the breaks in the rows in the series. So, for
    //     example, series[series_breaks[j]] would be the beginning of the
    //     (j + 1)th row, which would end at series[series_breaks[j + 1]].
    //     equal_length (const bool&): Whether the rows are all of equal
    //     time length.
    //     n_dim (const size_t&): The number of channels in the rows.

    size_t n_rows { series_breaks.size() - 1 };

    if (equal_length)
    {
        ssize_t n_t {
            (series_breaks[1] - series_breaks[0])
            / static_cast<ssize_t>(n_dim)
        };
        std::vector<ssize_t> shape {
            static_cast<ssize_t>(n_rows),
            static_cast<ssize_t>(n_dim),
            n_t
        };
        py::array_t<float> rtn_series_array { shape, series.data() };
        return rtn_series_array;
    }
    else
    {
        std::vector<py::array_t<float>> rtn_series_list {};

        for(size_t i_row = 0; i_row < n_rows; i_row++)
        {
            ssize_t n_t {
                (series_breaks[i_row + 1] - series_breaks[i_row])
                / static_cast<ssize_t>(n_dim)
            };
            std::vector<ssize_t> shape { static_cast<ssize_t>(n_dim), n_t };
            rtn_series_list.push_back(
                py::array_t<float> {
                    shape,
                    series.data() + series_breaks[i_row],
                }
            );
        }

        return py::cast(rtn_series_list);
    }
}
