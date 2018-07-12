import yaml
import os
import numpy as np

from hyperopt_conf import definition


class ConfigGenerator(object):
    def __init__(self, trainer_config_path=None):
        if trainer_config_path:
            self.trainer_config_path = trainer_config_path
        else:
            DEFAULT_CONF_PATH = 'trainer_config.yaml'
            self.trainer_config_path = DEFAULT_CONF_PATH


    def generate(self, env_name, params, output_file_name, params_dict_format=True):
        '''
        Use default config file and override it using the params.
        Create the modified yaml config file.
        If params_dict_format=True the params is like
            {
              "learning_rate": 0.001,
              "num_epoch": 2
            }
        else
            [[0.001, 2]]
        '''
        output_conf_path = 'configs/' + output_file_name + '.yaml'
        config_data_root = yaml.load(open(self.trainer_config_path))

        if env_name in config_data_root:
            config_data = config_data_root[env_name]
        else:
            config_data = config_data_root['default']

        if params_dict_format:
            for key, value in params.items():
                config_data[key] = value
        else:
            params = params[0]
            for i, variable in enumerate(definition):
                value = params[i]
                if variable['type'] == 'discrete':
                    value = np.int(value)
                config_data[variable['name']] = value


        os.makedirs(os.path.dirname(output_conf_path), exist_ok=True)
        with open(output_conf_path, 'w') as output_file:
            yaml.dump(config_data_root, output_file, default_flow_style=False)

        return output_conf_path
