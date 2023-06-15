import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


class DualRegressionModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "camembert/camembert-base",
        loss_aggreatation: str = "mean",
    ):
        """
        This class instantiates the pre-training model.
        :param model_name_or_path: The name or path of the model to be used for pre-training.
        """

        super().__init__()
        if "bart" in model_name_or_path:
            self.model = AutoModel.from_pretrained(
                model_name_or_path, output_hidden_states=True
            )
            self.model = self.model.encoder
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name_or_path, output_hidden_states=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.loss_aggreatation = loss_aggreatation

        # create two different regression heads for two tasks (latitude and longitude)
        self.lat_regression_head = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.long_regression_head = torch.nn.Linear(self.model.config.hidden_size, 1)

        self.crierion = torch.nn.MSELoss()

    def forward(
        self,
        batch,
    ):
        """
        This function is called to compute the loss for the specified task.
        :param batch: The batch of data.
        """
        predict = not batch.keys() & {"longitude", "latitude"}

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if not predict:
            latitudes = batch["latitude"]
            longitudes = batch["longitude"]

        # get the last hidden state
        last_hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).hidden_states[-1][:, 0, :]

        lat_predictions = self.lat_regression_head(last_hidden_state)
        long_predictions = self.long_regression_head(last_hidden_state)

        result = {"latitude": lat_predictions, "longitude": long_predictions}

        if not predict:
            lat_loss = self.crierion(lat_predictions.squeeze(), latitudes)
            long_loss = self.crierion(long_predictions.squeeze(), longitudes)

            if self.loss_aggreatation == "mean":
                loss = (lat_loss + long_loss) / 2
            elif self.loss_aggreatation == "sum":
                loss = lat_loss + long_loss
            else:
                raise ValueError("Only mean and sum are supported for loss aggregation")
            result |= {"loss": loss}

        return result

    def save_model(self, path):
        """
        This function is called to save the model to a specified path. E.g. "model.pt"
        :param path: The path where the model is saved.
        """

        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        This function is called to load the model.
        :param path: The path where the model is saved. E.g. "model.pt"
        """

        # load the state dict
        self.load_state_dict(torch.load(path))
