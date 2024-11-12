import hydra
from omegaconf import DictConfig,OmegaConf


@hydra.main(config_path="configs/", config_name="test.yaml", version_base="1.2.0")
def main(config: DictConfig):
    
    from src.test import run
    run(config)


if __name__ == "__main__":
    main()