from tensorboard.backend.event_processing import event_accumulator
import os
import glob

class SummariesReader(object):
    def __init__(self, name):
        last_event_path = get_latest_file('./summaries/' + name + '/*')
        print(last_event_path)
        self.event = event_accumulator.EventAccumulator(last_event_path, size_guidance={event_accumulator.SCALARS:0});
        self.event.Reload();

    def get_scalar_keys(self):
        return self.event.scalars.Keys()

    def get_scalar(self, scalar_key):
        return self.event.Scalars(scalar_key)


# https://codereview.stackexchange.com/a/120500
def get_latest_file(path, *paths):
    """Returns the name of the latest (most recent) file
    of the joined path(s)"""
    fullpath = os.path.join(path, *paths)
    list_of_files = glob.glob(fullpath)
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


if __name__ == '__main__':
    sr = SummariesReader('ppo')

    print(sr.get_scalar_keys())
    print(sr.get_scalar('Info/cumulative_reward'))
