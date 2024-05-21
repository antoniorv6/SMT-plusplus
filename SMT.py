import numpy as np

import torch
import wandb
import torch.nn as nn
import lightning.pytorch as L

from torchinfo import summary

from config_typings import SMTConfig

from model.ConvEncoder import Encoder
from model.ConvNextEncoder import ConvNextEncoder
from model.Decoder import Decoder
from model.PositionEncoding import PositionalEncoding2D, PositionalEncoding1D

from eval_functions import compute_poliphony_metrics

class SMT(L.LightningModule):
    def __init__(self, config:SMTConfig, w2i, i2w) -> None:
        super().__init__()
        
        if config.encoder_type == "NexT":
            self.encoder = ConvNextEncoder(in_chans=config.in_channels, depths=[3,3,9], dims=[64, 128, 256])
        else:
            self.encoder = Encoder(in_channels=config.in_channels)

        self.decoder = Decoder(config.d_model, config.dim_ff, config.num_dec_layers, config.max_len + 1, len(w2i))
        self.positional_2D = PositionalEncoding2D(config.d_model, (config.max_height//16) + 1, (config.max_width//8) + 1)

        self.padding_token = 0

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token)

        self.valpredictions = []
        self.valgts = []

        self.w2i = w2i
        self.i2w = i2w

        self.maxlen = config.max_len

        #self(torch.randn(1,1,config.max_height,config.max_width).to(torch.device("cuda")), torch.randint(0, len(w2i), (1,config.max_len)).to(torch.device("cuda")))
        #import sys
        #sys.exit()
        self.worst_loss_image = None
        self.worst_training_loss = -1
        summary(self, input_size=[(1,1,config.max_height,config.max_width), (1,config.max_len)], 
                dtypes=[torch.float, torch.long])

        self.save_hyperparameters()

    def forward(self, x, y_pred):
        encoder_output = self.encoder(x)
        b, c, h, w = encoder_output.size()
        reduced_size = [s.shape[:2] for s in encoder_output]
        ylens = [len(sample) for sample in y_pred]
        cache = None

        pos_features = self.positional_2D(encoder_output)
        features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(2,0,1)
        enhanced_features = features
        enhanced_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2,0,1)
        output, predictions, _, _, weights = self.decoder(features, enhanced_features, y_pred[:, :-1], reduced_size, 
                                                           [max(ylens) for _ in range(b)], encoder_output.size(), 
                                                           start=0, cache=cache, keep_all_weights=True)
    
        return output, predictions, cache, weights


    def forward_encoder(self, x):
        return self.encoder(x)
    
    def forward_decoder(self, encoder_output, last_preds, cache=None):
        b, c, h, w = encoder_output.size()
        reduced_size = [s.shape[:2] for s in encoder_output]
        ylens = [len(sample) for sample in last_preds]
        cache = cache

        pos_features = self.positional_2D(encoder_output)
        features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(2,0,1)
        enhanced_features = features
        enhanced_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2,0,1)
        output, predictions, _, _, weights = self.decoder(features, enhanced_features, last_preds[:, :], reduced_size, 
                                                           [max(ylens) for _ in range(b)], encoder_output.size(), 
                                                           start=0, cache=cache, keep_all_weights=True)
    
        return output, predictions, cache, weights
    
    def configure_optimizers(self):
        return torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4, amsgrad=False)

    def training_step(self, train_batch):
        x, di, y = train_batch
        output, predictions, cache, weights = self.forward(x, di)
        loss = self.loss(predictions, y[:, :-1])
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        if loss > self.worst_training_loss:
            self.worst_loss_image = x
            self.worst_training_loss = loss
        
        return loss

    def on_train_epoch_end(self):
        #plot the worst training loss image in wandb
        self.logger.experiment.log({"worst_training_loss_image": [wandb.Image(self.worst_loss_image.squeeze(0).cpu().numpy())]})
        self.worst_training_loss = -1
        self.worst_loss_image = None

    def validation_step(self, val_batch, batch_idx):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x, _, y = val_batch
        encoder_output = self.forward_encoder(x)
        predicted_sequence = torch.from_numpy(np.asarray([self.w2i['<bos>']])).to(device).unsqueeze(0)
        cache = None
        for i in range(self.maxlen):
             output, predictions, cache, weights = self.forward_decoder(encoder_output, predicted_sequence.long(), cache=cache)
             predicted_token = torch.argmax(predictions[:, :, -1]).item()
             predicted_sequence = torch.cat([predicted_sequence, torch.argmax(predictions[:, :, -1], dim=1, keepdim=True)], dim=1)
             if predicted_token == self.w2i['<eos>']:
                 break
        
        dec = "".join([self.i2w[token.item()] for token in predicted_sequence.squeeze(0)[1:]])
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join([self.i2w[token.item()] for token in y.squeeze(0)[:-1]])
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.valpredictions.append(dec)
        self.valgts.append(gt)
    
    def on_validation_epoch_end(self, name="val"):
        cer, ser, ler = compute_poliphony_metrics(self.valpredictions, self.valgts)
        
        random_index = np.random.randint(0, len(self.valpredictions))
        predtoshow = self.valpredictions[random_index]
        gttoshow = self.valgts[random_index]
        print(f"[Prediction] - {predtoshow}")
        print(f"[GT] - {gttoshow}")

        self.log(f'{name}_CER', cer, prog_bar=True)
        self.log(f'{name}_SER', ser, prog_bar=True)
        self.log(f'{name}_LER', ler, prog_bar=True)

        self.valpredictions = []
        self.valgts = []

        return ser
    
    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)
    
    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(name="test")
        