from smt_trainer import SMTPP_Trainer

model_wrapper = SMTPP_Trainer(maxh=2512, maxw=2512, maxlen=5512, out_categories=2100, 
                                  padding_token=0, in_channels=1, w2i={}, i2w={}, 
                                  d_model=256, dim_ff=256, num_dec_layers=8)

