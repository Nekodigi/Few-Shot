import hydra


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    print(cfg)
    cfg.base_model = "test"
    print(cfg.version)


if __name__ == "__main__":
    main()


# TODO AFTER THAT POSSIBLY BREAK DOWN INTO SMALLER PATCH TO CATCH LOCAL.
