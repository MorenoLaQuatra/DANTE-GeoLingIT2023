from transformers import AutoModelForMaskedLM, AutoModel
from transformers import AutoTokenizer
import torch
import numpy as np
from torch import nn

class ModelForPretraining(nn.Module):

    def __init__(
        self,
        model_name_or_path: str = "camembert/camembert-base",
        mlm: bool = True,
        nsp: bool = True,
        dc: bool = True,
        tc: bool = True,
        num_labels_dc: int = 22,
    ):
        '''
        This class instantiates the pre-training model.
        :param model_name_or_path: The name or path of the model to be used for pre-training.
        :param mlm: Whether to use the masked language modeling task or not.
        :param nsp: Whether to use the next sentence prediction task or not.
        '''

        super().__init__()
        if "bart" in model_name_or_path:
            self.model = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True)
            self.model = self.model.encoder
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, output_hidden_states=True)
        
        self.model_name_or_path = model_name_or_path
        print(f"Model name or path: {model_name_or_path}")
        print(f"Number of parameters: {torch.sum(torch.tensor([p.numel() for p in self.model.parameters()]))}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.mlm = mlm
        self.nsp = nsp
        self.dc = dc
        self.tc = tc
        self.num_labels_dc = num_labels_dc

        if self.nsp:
            # linear layer for the nsp task
            self.nsp_classifier = torch.nn.Linear(self.model.config.hidden_size, 1)
            # nsp criterion is Binary Cross Entropy
            self.nsp_criterion = torch.nn.BCEWithLogitsLoss()

        if self.mlm:
            if "bart" in model_name_or_path:
                self.mlm_classifier = torch.nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size)
            # mlm criterion is Cross Entropy
            self.mlm_criterion = torch.nn.CrossEntropyLoss()

        if self.dc:
            # linear layer for the dc task 
            self.dc_classifier = torch.nn.Linear(self.model.config.hidden_size, self.num_labels_dc)
            # dc criterion is Cross Entropy
            self.dc_criterion = torch.nn.CrossEntropyLoss()

        if self.tc:
            # linear layer for the token classification task
            self.tc_classifier = torch.nn.Linear(self.model.config.hidden_size, self.num_labels_dc)
            # tc criterion is Cross Entropy
            self.tc_criterion = torch.nn.CrossEntropyLoss()




    def next_sentence_prediction_task(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        '''
        This function is called to compute the next sentence prediction loss.
        :param input_ids: The tokenized text.
        :param attention_mask: The attention mask.
        :param labels: The labels for the next sentence prediction task.
        :return: The next sentence prediction loss.
        '''

        # get the last hidden state of the first token (CLS)
        last_hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).hidden_states[-1][:, 0, :]

        # get the logits for the nsp task
        logits = self.nsp_classifier(last_hidden_state)
        logits = logits.squeeze(-1)

        # compute the loss (cross entropy)
        loss = self.nsp_criterion(logits, labels)

        return loss

    def document_classification_task( # dialect classification task
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        '''
        This function is called to compute the next sentence prediction loss.
        :param input_ids: The tokenized text.
        :param attention_mask: The attention mask.
        :param labels: The labels for the next sentence prediction task.
        :return: The next sentence prediction loss.
        '''

        # get the last hidden state of the first token (CLS)
        last_hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).hidden_states[-1][:, 0, :]

        # get the logits for the dc task
        logits = self.dc_classifier(last_hidden_state)

        logits = logits.squeeze(-1)

        # compute the loss
        loss = self.dc_criterion(logits, labels)

        return loss

    def masked_language_modeling_task(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        '''
        This function is called to compute the masked language modeling loss.
        :param input_ids: The tokenized text.
        :param attention_mask: The attention mask.
        :param labels: The labels for the masked tokens.
        :return: The masked language modeling loss.
        '''
        
        if "bart" in self.model_name_or_path:
            # get the last hidden state of the first token (CLS)
            predicted_ids = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).hidden_states[-1]
            predicted_ids = self.mlm_classifier(predicted_ids)
        else:
            # get the predicted ids for all tokens
            predicted_ids = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        # compute the loss
        loss = self.mlm_criterion(predicted_ids.view(-1, predicted_ids.size(-1)), labels.view(-1))

        return loss

    def token_classification_task(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        '''
        This function is called to compute the token classification loss.
        :param input_ids: The tokenized text.
        :param attention_mask: The attention mask.
        :param labels: The labels for the tokens.
        :return: The token classification loss.
        '''

        # get the predicted ids for all tokens
        predicted_ids = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).hidden_states[-1]

        # apply the TC layer
        predicted_ids = self.tc_classifier(predicted_ids)

        # compute the loss
        loss = self.tc_criterion(predicted_ids.view(-1, predicted_ids.size(-1)), labels.view(-1))

        return loss

    def forward(
        self,
        batch,
        current_step,
    ):
        '''
        This function is called to compute the loss for the specified task.
        :param input_ids: The tokenized text.
        :param attention_mask: The attention mask.
        :param labels: The labels for the specified task.
        :param task: The task for which the loss is computed.
        :return: The loss for the specified task.
        '''

        task_index = batch['task_index'][0].item()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        if task_index == 0: # mlm task
            loss = self.masked_language_modeling_task(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        elif task_index == 1: # nsp task
            loss = self.next_sentence_prediction_task(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        elif task_index == 2: # dc task
            loss = self.document_classification_task(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        elif task_index == 3: # tc task
            loss = self.token_classification_task(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            raise ValueError('Invalid task index.')

        return loss


    def save_model(self, path):
        '''
        This function is called to save the model.
        :param path: The path where the model is saved.
        '''

        self.model.save_pretrained(path)
