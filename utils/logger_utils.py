from pytorch_lightning.loggers import  Logger


class PrintLogger(Logger):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return "PrintLogger"

    @property
    def version(self):
        return "0.1"

    def log_hyperparams(self, params):
        print("Hyperparameters:")
        print(params)

    def log_metrics(self, metrics, step):
        print(metrics)

    def experiment(self):
        # This logger does not create an experiment object.
        return None
