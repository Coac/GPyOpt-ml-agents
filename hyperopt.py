import GPyOpt
from docopt import docopt

import hyperopt_conf
from train_runner import TrainRunner

if __name__ == '__main__':
    _USAGE = '''
    Usage:
      hyperopt (<env>)
    '''
    options = docopt(_USAGE)

    train_runner = TrainRunner(options['<env>'])

    bayesian_opt = GPyOpt.methods.BayesianOptimization(train_runner.f,
                                                       domain=hyperopt_conf.definition,
                                                       acquisition_type='EI',
                                                       normalize_Y=True,
                                                       initial_design_numdata=hyperopt_conf.batch_size,
                                                       evaluator_type='local_penalization',
                                                       batch_size=hyperopt_conf.batch_size,
                                                       num_cores=hyperopt_conf.num_cores,
                                                       acquisition_jitter=0,
                                                       maximize=True)

    bayesian_opt.run_optimization(hyperopt_conf.max_iter)

    print('Best hyperparameters:', bayesian_opt.x_opt)
    print('Max reward:', bayesian_opt.fx_opt)
