definition = [{
  'name': 'learning_rate',
  'type': 'continuous',
  'domain': [1e-5, 1e-3]
}, {
  'name': 'epsilon',
  'type': 'continuous',
  'domain': [0.1, 0.3]
}, {
  'name': 'gamma',
  'type': 'continuous',
  'domain': [0.8, 0.995]
}, {
  'name': 'lambd',
  'type': 'continuous',
  'domain': [0.9, 0.95]
}, {
  'name': 'num_epoch',
  'type': 'discrete',
  'domain': [3, 10]
}, {
  'name': 'beta',
  'type': 'continuous',
  'domain': [1e-4, 1e-2]
}, {
  'name': 'beta',
  'type': 'continuous',
  'domain': [1e-4, 1e-2]
}, {
  'name': 'num_layers',
  'type': 'discrete',
  'domain': [1, 3]
}, {
  'name': 'hidden_units',
  'type': 'discrete',
  'domain': [32, 64, 128, 256, 512]
}]

batch_size = 8
num_cores = 52
max_iter = 16
