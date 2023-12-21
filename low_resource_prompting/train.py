import hydra
import lightning as pl
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="train.yaml", version_base=None)
def train(cfg: DictConfig) -> None:
    # set random seed
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)

    # set up data module
    print(f"instantiating {cfg.data._target_}")
    data_module = hydra.utils.instantiate(cfg.data)  # intaiate creates an object

    # set up model
    print(f"instantiating {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)

    # set up trainer
    print(f"instantiating {cfg.trainer._target_}")
    trainer = hydra.utils.instantiate(cfg.trainer)

    # train model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
