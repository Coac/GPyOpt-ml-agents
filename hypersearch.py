import os
import subprocess
import threading
import signal
import sys
from multiprocessing.dummy import Pool as ThreadPool
import portpicker

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

    for worker_id in range(10):
        unused_port = portpicker.pick_unused_port()
        proc = subprocess.Popen(['python', 'learn.py', 'test123.x86_64', '--train', '--worker-id=' + str(unused_port)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
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
