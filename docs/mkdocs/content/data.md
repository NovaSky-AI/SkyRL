# Data Interface

Our interface for training data is modelled after [DataProto](https://verl.readthedocs.io/en/latest/api/data.html) in VERL but is much simpler.

## Trainer APIs

::: skyrl.backends.skyrl_train.training_batch.TensorBatch
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.backends.skyrl_train.training_batch.TrainingInput
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.backends.skyrl_train.training_batch.TrainingInputBatch
    options:
      show_root_heading: true
      members_order: source

::: skyrl.backends.skyrl_train.training_batch.TrainingOutputBatch
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Generator APIs

::: skyrl.train.generators.base.GeneratorInput
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.generators.base.GeneratorOutput
    options:
      show_root_heading: true
      members_order: source
      show_bases: true
