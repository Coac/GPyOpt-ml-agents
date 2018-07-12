import os
import subprocess
import threading
import signal
import sys
from multiprocessing.dummy import Pool as ThreadPool
import portpicker
from train_runner import TrainRunner
from collections import Mapping
from functools import partial, reduce
import operator
from itertools import product
from docopt import docopt

from config_generator import ConfigGenerator


# ---------- From scikit-learn source: https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/model_selection/_search.py#L48
class ParameterGrid(object):
    """Grid of parameters with a discrete number of values for each.
    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    Read more in the :ref:`User Guide <search>`.
    Parameters
    ----------
    param_grid : dict of string to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.
        An empty dict signifies default parameters.
        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.
    Examples
    --------
    >>> from sklearn.model_selection import ParameterGrid
    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
    >>> list(ParameterGrid(param_grid)) == (
    ...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
    ...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True
    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
    ...                               {'kernel': 'rbf', 'gamma': 1},
    ...                               {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True
    See also
    --------
    :class:`GridSearchCV`:
        Uses :class:`ParameterGrid` to perform a full parallelized parameter
        search.
    """

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration
        Parameters
        ----------
        ind : int
            The iteration index
        Returns
        -------
        params : dict of string to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')

# ----------

def output_reader(proc):
    for line in iter(proc.stdout.readline, b''):
        print('[{0}] {1}'.format(proc.pid, line.decode('utf-8')), end='')

def terminate_proc(sig, proc):
    proc.send_signal(sig)
    try:
        outs, _ = proc.communicate(timeout=10)
        print('-- [{0}]'.format(proc.pid), 'subprocess exited with return code', proc.returncode)
        print('[{0}]'.format(proc.pid), outs.decode('utf-8'))
    except subprocess.TimeoutExpired:
        print('-- [{0}]'.format(proc.pid), 'subprocess did not terminate in time')

def grid_search(env_name, params_grid):
    conf_gen = ConfigGenerator()
    procs = []

    train_runner = TrainRunner(env_name)

    params_grid = list(ParameterGrid(params_grid))

    for params in params_grid:
        run_id = str(params).strip("{}").replace(': ', '').replace('\'', '').replace(', ', '_')
        conf_path = conf_gen.generate(env_name, params, run_id, params_dict_format=True)
        proc = train_runner.start_train_process(conf_path, run_id)
        procs.append(proc)

    def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            for proc in procs:
                terminate_proc(sig, proc)
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    pool = ThreadPool(len(procs))
    pool.map(output_reader, procs)

    pool.close()
    pool.join()

if __name__ == '__main__':
    _USAGE = '''
    Usage:
      grid_search (<env>)
    '''
    options = docopt(_USAGE)
    env_name = options['<env>']

    params_grid = {
        'learning_rate': [0.01, 0.02],
        'num_epoch': [1, 2]
    }

    grid_search(env_name, params_grid)
