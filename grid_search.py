import signal
import subprocess
import sys
from multiprocessing.dummy import Pool as ThreadPool

from docopt import docopt

from config_generator import ConfigGenerator
from grid_search_conf import params_grid
from parameter_grid import ParameterGrid
from train_runner import TrainRunner


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

def grid_search(env_name, params_grid, load):
    conf_gen = ConfigGenerator()
    procs = []

    train_runner = TrainRunner(env_name)

    params_grid = list(ParameterGrid(params_grid))

    for params in params_grid:
        run_id = str(params).strip("{}").replace(': ', '').replace('\'', '').replace(', ', '_')
        conf_path = conf_gen.generate(env_name, params, run_id, params_dict_format=True)
        proc = train_runner.start_train_process(conf_path, run_id, load)
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
          grid_search <env> [options]
          grid_search --help

        Options:
            --load                     Whether to load the model or randomly initialize [default: False].
        '''
    options = docopt(_USAGE)
    env_name = options['<env>']

    load = options['--load']
    if load:
        load = '--load'
    else:
        load = ''

    grid_search(env_name, params_grid, load)
