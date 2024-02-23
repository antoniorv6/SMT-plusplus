import gin

import numpy as np

import torch
import torch.nn as nn
import lightning.pytorch as L

from model.ConvEncoder import Encoder
from model.ConvNextEncoder import ConvNextEncoder
from model.Decoder import Decoder
from model.PositionEncoding import PositionalEncoding2D
from torchinfo import summary
from eval_functions import compute_poliphony_metrics


@gin.configurable
class SMT(L.LightningModule):
    def __init__(self, maxh, maxw, maxlen, out_categories, padding_token, in_channels, w2i, i2w, out_dir, 
                 d_model=None, dim_ff=None, num_dec_layers=None, encoder_type="Normal") -> None:
        super().__init__()
        
        if encoder_type == "NexT":
            self.encoder = ConvNextEncoder(in_chans=1, depths=[3,3,9], dims=[64, 128, 256])
        else:
            self.encoder = Encoder(in_channels=in_channels)

        self.decoder = Decoder(d_model, dim_ff, num_dec_layers, maxlen, out_categories)
        self.positional_2D = PositionalEncoding2D(d_model, maxh, maxw)

        self.padding_token = padding_token

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token)

        self.valpredictions = []
        self.valgts = []

        self.w2i = w2i
        self.i2w = i2w
        self.maxlen = maxlen
        self.out_dir=out_dir

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
        return loss

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

def get_SMT_network(in_channels, max_height, max_width, max_len, out_categories, w2i, i2w, out_dir, model_name=None):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMT(in_channels=in_channels, maxh=(max_height//16)+1, maxw=(max_width//8)+1, 
                maxlen=max_len+1, out_categories=out_categories, 
                padding_token=0, w2i=w2i, i2w=i2w, out_dir=out_dir).to(device)
    
    #with torch.no_grad():
    #    print(max_height, max_width, max_len)
    #    _ = model(torch.randn((1,1,max_height,max_width), device=device), torch.randint(low=0, high=100,size=(1,max_len), device=device).long())
    #import sys
    #sys.exit()
    summary(model, input_size=[(1,1,max_height,max_width), (1,max_len)], dtypes=[torch.float, torch.long])

    return model
