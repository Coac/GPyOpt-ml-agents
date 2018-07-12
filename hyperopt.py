import GPy
import GPyOpt
import hyperopt_conf
from train_runner import TrainRunner

if __name__ == '__main__':
    train_runner = TrainRunner('3DBall')

    bayesian_opt = GPyOpt.methods.BayesianOptimization(train_runner.f,
                                                    domain=hyperopt_conf.definition,
                                                    acquisition_type = 'EI',
                                                    normalize_Y = True,
                                                    initial_design_numdata = hyperopt_conf.batch_size,
                                                    evaluator_type = 'local_penalization',
                                                    batch_size = hyperopt_conf.batch_size,
                                                    num_cores = hyperopt_conf.num_cores,
                                                    acquisition_jitter = 0,
                                                    maximize=True)

    bayesian_opt.run_optimization(hyperopt_conf.max_iter)

    print('x_opt', bayesian_opt.x_opt)
    print('fx_opt', bayesian_opt.fx_opt)
