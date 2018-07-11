import os
import subprocess
import threading
import signal
import sys
from multiprocessing.dummy import Pool as ThreadPool
import portpicker
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

if __name__ == '__main__':
    procs = []

    train_runner = TrainRunner('test123')

    # TODO: grid search
    for worker_id in range(10):
        proc = train_runner.start_train_process('', 'ppo')
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
