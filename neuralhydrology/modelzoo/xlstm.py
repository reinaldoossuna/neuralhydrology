import logging
from typing import Dict
from dacite import from_dict
from dacite import Config as DaciteConfig

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)

try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
    )
except ImportError:
    LOGGER.info("xlstm not found, xlstm models will not work")


class xLSTM(BaseModel):
    """xLSTM model, which relies on https://github.com/NX-AI/xlstm.

    This class implements the xLSTM with a combined model head, as specified in the config file, and a transition layer to ensure the input dimensions match the xlstm specifications. Please read the xlstm documentation in github to better learn about the required hyperparameters.

    if one set values for context_length and embedding_dim, this will be updated with the values of seq_length and hidden_size respectively.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ["embedding_net", "transition_layer", "xlstm", "head"]

    def __init__(self, cfg: Config):
        super(xLSTM, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        # using a linear layer to move from the emdedded_layer dims to the specified hidden size
        self.transition_layer = nn.Linear(self.embedding_net.output_size, self.cfg.hidden_size)

        xlstm_config = from_dict(
            xLSTMBlockStackConfig, cfg.xlstm_config, config=DaciteConfig(strict=True)
        )
        self.xlstm = xLSTMBlockStack(xlstm_config)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the CudaLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)

        # reshaping dimensions to what xlstm expects:
        x_d_transition = self.transition_layer(x_d)

        output = self.xlstm(x_d_transition)

        # reshape to [batch_size, seq, n_hiddens]
        output = output.transpose(0, 1)

        pred = {"y_hat": output}
        pred.update(self.head(self.dropout(output)))
        return pred
