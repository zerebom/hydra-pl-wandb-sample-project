# hydra-pl-wandb-sample-project

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