# hydra-pl-wandb-sample-project
hydra-pl-wandb-sample-project is a NN experiment management code using hydra, pytorch-lightinig, and wandb.　　

blog: https://zerebom.hatenablog.com/entry/2020/12/11/174527

### directory
```
.
├── config
│   ├── callbacks
│   │   └── default_callbacks.yaml
│   ├── data
│   │   └── default_data.yaml
│   ├── default_config.yaml
│   ├── env
│   │   └── default_env.yaml
│   ├── logger
│   │   └── wandb_logger.yaml
│   ├── model
│   │   └── autoencoder.yaml
│   └── trainer
│       └── default_trainer.yaml
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── README.md
└── src
    ├── factory
    │   ├── dataset.py
    │   ├── logger.py
    │   └── networks
    │       └── autoencoder.py
    └── train.py
```

### requirements
python 3.8  
poetry

### usage

```
poetry install
poetry run python train.py
```
