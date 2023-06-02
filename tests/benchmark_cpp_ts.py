import argparse
import time

import numpy as np
import torchcast as tc


def python_load_ts_file(series_file: str):
    '''
    Parses a .ts file from the scikit-time package. This is documented at:

        https://github.com/alan-turing-institute/sktime/blob/main/examples/loading_data.ipynb

    This returns a batch of series, in the form of a 3-dimensional
    :class:`torch.Tensor` in NCT arrangement.
    '''
    if isinstance(series_file, str):
        with open(series_file, 'r') as series_file:
            return python_load_ts_file(series_file)

    # Valid keys in the ts header:
    #
    # problemName (str)
    # timeStamps (bool)
    # missing (bool)
    # univariate (bool)
    # dimensions (int)
    # equalLength (bool)
    # seriesLength (int)
    # classLabel (bool, followed by Sequence[str])

    in_header, header, values, labels = True, {}, [], []

    for line in series_file.readlines():
        # Remove '\n' from line
        line = line.strip('\n')
        if in_header:
            # ts files allow comments
            if line.startswith('#'):
                continue

            elif line.startswith('@data'):
                is_sparse = (header.get('@timeStamps', 'false') == 'true')
                n_t = int(header.get('@seriesLength', '0'))
                if header.get('@univariate', 'false') == 'true':
                    n_dim = 1
                else:
                    n_dim = int(header['@dimensions'])

                categories = header.get('@classLabel', 'false')
                use_labels, *category_names = categories.split()
                use_labels = (use_labels == 'true')
                in_header = False

            elif line.startswith('@'):
                key, value = line.split(' ', 1)
                header[key] = value

            else:
                raise ValueError(f'Cannot parse: {line}')

        else:
            if use_labels:
                # When labels are present, they are appended to the end of the
                # line following a ':'.
                line, label = line.rsplit(':', 1)
                labels.append(category_names.index(label))

            if is_sparse:
                value = [
                    np.full((n_t,), np.nan, dtype=np.float32)
                    for _ in range(n_dim)
                ]
                for dim, row in enumerate(line.split(':')):
                    for entry in row.strip('()').split('),('):
                        t, v = entry.split(',')
                        value[dim][int(t)] = float(v)

            else:
                value = [
                    [np.array(x, dtype=np.float32) for x in row.split(',')]
                    for row in line.replace('?', 'nan').split(':')
                ]

            values.append(value)

    if not n_t:
        n_t = max(max(len(row) for row in value) for value in values)

    series = np.full((len(values), n_dim, n_t), np.nan, dtype=np.float32)
    for i, line in enumerate(values):
        for j, row in enumerate(line):
            series[i, j, :len(row)] = row

    if use_labels:
        return series, labels
    else:
        return series


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--num-runs', type=int, default=10)
    args = parser.parse_args()

    start_time = time.perf_counter()
    for _ in range(args.num_runs):
        python_load_ts_file(args.path)
    python_elapsed_time = (time.perf_counter() - start_time) / args.num_runs

    start_time = time.perf_counter()
    for _ in range(10):
        tc.load_ts_file(args.path)
    cpp_elapsed_time = (time.perf_counter() - start_time) / args.num_runs

    print(f'Python: {python_elapsed_time:.4f} sec')
    print(f'C++:    {cpp_elapsed_time:.4f} sec')
