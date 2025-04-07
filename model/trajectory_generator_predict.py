import pytorch_lightning as pl
from model.trajectory_generator_model import TrajectoryGenerator
from utils.config import Configuration


class TrajectoryPredictModule(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.model = TrajectoryGenerator(cfg)

    def forward(self, batch):
        """
        For inference, call the model's own predict() method to generate tokens.
        """
        # Use the configured max frame or token count for generation.
        return self.model.predict(batch, predict_token_num=self.cfg.max_frame)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This method is automatically called for each batch during the prediction loop.
        """
        predicted_tokens = self.forward(batch)
        return predicted_tokens


# Example usage:
# if __name__ == "__main__":
#     # Initialize your configuration
#     cfg = Configuration()
#
#     # Load your trained model checkpoint.
#     # You can use load_from_checkpoint if you saved the LightningModule using the trainer.
#     predictor = TrajectoryPredictModule.load_from_checkpoint("path/to/checkpoint.ckpt", cfg=cfg)
#
#     # Create a Lightning Trainer (adjust settings such as GPU usage as needed)
#     trainer = pl.Trainer(gpus=1)
#
#     # Prepare your dataloader for prediction. The dataloader should return a dictionary with
#     # the keys "input_ids", "ego_info", "agent_info", and "goal" (preprocessed as during training).
#     # For example:
#     # predict_dataloader = DataLoader(my_prediction_dataset, batch_size=..., shuffle=False)
#
#     # Run prediction using the Lightning Trainer
#     predictions = trainer.predict(predictor, dataloaders=predict_dataloader)
#
#     # Process or display predictions as needed.
#     for pred in predictions:
#         print("Predicted Tokens:", pred)
