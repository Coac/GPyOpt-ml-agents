import portpicker
import subprocess
from config_generator import ConfigGenerator

class TrainRunner(object):
    def __init__(self, env_name):
        self.env_name = env_name
        self.conf_gen = ConfigGenerator()


    def f(self, params):
        '''
        Function to optimize
        Runs a training process and wait for its termination
        '''
        conf_path = self.conf_gen.generate(params)

        run_id = 'ppo'
        proc = self.start_train_process(env_name, conf_path, run_id)
        proc.wait()

        reward = SummariesReader(run_id).get_scalar('Info/cumulative_reward')
        print('Youpi c\'est terminé')

        return reward

    # x^2 + y^3 + 5
    def f(self, params):
        return params[0][0]**2 + params[0][1] **3 + 5


    @staticmethod
    def start_train_process(env_name, conf_path):
        unused_port = portpicker.pick_unused_port()
        proc = subprocess.Popen(['python', 'learn.py', env_name, '--train', '--worker-id=' + str(unused_port), '--trainer-config-path=' + str(conf_path)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        return proc


if __name__ == '__main__':
    train_runner = TrainRunner('test123')
    reward = train_runner.f(params)
    print(reward)
