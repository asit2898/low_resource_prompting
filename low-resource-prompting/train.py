import hydra
import lightning as pl
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="train.yaml", version_base=None)
def train(cfg: DictConfig) -> None:
    # set random seed
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)

    # set up data module
    data_module = hydra.utils.instantiate(cfg.data)


if __name__ == "__main__":
    train()
