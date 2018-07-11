from hyperopt_conf import definition


class ConfigGenerator(object):
    def __init__(self):
        pass

    def generate(self, params):
        conf_path = 'trainer_config.yaml'

        print(definition)

        return conf_path
