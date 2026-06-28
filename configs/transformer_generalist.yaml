# Generalist transformer: trained on all three tasks, tested on all three.
# Same pipeline as the GRU generalist — only the model block differs.
output_dir: checkpoints/generalist

model:
  type: transformer          # key in MODELS (models.py); add new architectures there
  input_dim: 18     # 6 positions + 6 velocities + 6 accelerations (add_velocity + add_acceleration below)
  output_dim: 6     #two last heads: one for mus one for sigmas
  d_model: 64
  nhead: 4
  num_layers: 2
  dim_feedforward: 128
  dropout: 0.2
  max_len: 64
  causal: false             # true: attend only to past (matches GRU/TCN); false: full bidirectional encoder

data:
  train_task: R+M+L
  test_task: R+M+L   # overlapping tasks -> split_data auto-uses disjoint patients
  batch_size: 16384
  split_percentages: [0.7, 0.15, 0.15]
  cross_patients: false   # some patients do more tasks, if false the same patient can appear in train and test but with two diff tasks; if true, patients are disjoint across train and test
  sequence_length: 64     # input window = sequence_length frames, stride 2 (spans 2x frames)
  time_augmentation: false
  neg_augmentation: true
  add_velocity: true       # append first-difference (velocity) channels
  add_acceleration: true   # append second-difference (acceleration) channels -> input_dim 18


train:
  loss: gaussian_nll
  beta: 0.5
  epochs: 150
  patience: 100
  lr: 1.0e-3
  weight_decay: 1.0e-4
