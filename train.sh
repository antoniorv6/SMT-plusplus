#!/bin/bash

python train.py +cl=base_cl +data=mozarteum_bekern +experiment=mozarteum-finetuning +model_setup=SMT_Next +experiment.pretrain_weights=weights/Mozarteum/SMT_NexT_Mozarteum_pretraining.ckpt +metadata.corpus_name=Mozarteum +metadata.model_name=SMT_Next_Mozarteum_FP +data.fold=1 +cl.skip_progressive=False +cl.skip_cl=False