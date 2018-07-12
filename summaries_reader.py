import glob
import os

from tensorboard.backend.event_processing import event_accumulator


class SummariesReader(object):
    def __init__(self, name):
        last_event_path = get_latest_file('./summaries/' + name + '/*')
        self.event = event_accumulator.EventAccumulator(last_event_path, size_guidance={event_accumulator.SCALARS: 0})
        self.event.Reload()

    def get_scalar_keys(self):
        return self.event.scalars.Keys()

    def get_scalar(self, scalar_key):
        return self.event.Scalars(scalar_key)


# https://codereview.stackexchange.com/a/120500
def get_latest_file(path, *paths):
    """Returns the name of the latest (most recent) file
    of the joined path(s)"""
    full_path = os.path.join(path, *paths)
    list_of_files = glob.glob(full_path)
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


if __name__ == '__main__':
    sr = SummariesReader('2018-07-11_13-57-55.645635')

    print(sr.get_scalar_keys())
    print(sr.get_scalar('Info/cumulative_reward')[-1].value)
