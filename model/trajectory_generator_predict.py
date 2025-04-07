from model.trajectory_generator_model import TrajectoryGenerator
from utils.config import Configuration, InferenceConfiguration
from utils.trajectory_utils import detokenize_traj_waypoints
import pytorch_lightning as pl
import torch


class TrajectoryPredictModule(pl.LightningModule):
    def __init__(self, infer_cfg: InferenceConfiguration, train_cfg: Configuration, device: str):
        super().__init__()
        self.inf_cfg = infer_cfg
        self.train_cfg = train_cfg
        self.infer_device = torch.device('cuda') if device == "gpu" else torch.device('cpu')
        self.model = TrajectoryGenerator(self.train_cfg)
        self.load_model(self.inf_cfg.model_ckpt_path)

    def load_model(self, ck_path):
        ckpt = torch.load(ck_path, weights_only=False)
        state_dict = {k.replace('gen_model.', ''): v for k, v in ckpt['state_dict'].items()}
        print(state_dict.keys())
        self.model.load_state_dict(state_dict)
        self.model.to(self.infer_device)
        self.model.eval()

    def forward(self, batch):
        """
        For inference, call the model's own predict() method to generate tokens.
        """
        # Use the configured max frame or token count for generation.
        return self.model.predict(batch, predict_token_num=self.cfg.max_frame)

    def inference_step(self, batch):
        """
        This method is automatically called for each batch during the prediction loop.
        """
        predicted_tokens = self.forward(batch)
        return predicted_tokens

    def inference_transformer(self, data):
        pred_traj_point, _, _ = self.model.predict(data, predict_token_num=self.cfg.max_frame)
        pred_traj_point_update = pred_traj_point[0][1:]
        pred_traj_point_update = self.remove_invalid_content(pred_traj_point_update)

        delta_predicts = detokenize_traj_waypoints(
            pred_traj_point_update,
            self.cfg.token_nums,
            self.cfg.item_number
        )

        return delta_predicts

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
