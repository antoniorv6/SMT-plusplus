import torch
from torch.utils.data import DataLoader
from data import RealDataset, batch_preparation_img2seq
from smt_trainer import SMTPP_Trainer
from eval_functions import compute_poliphony_metrics

torch.set_float32_matmul_precision('high')

def main():
    data = RealDataset(data_path=f"Data/Mozarteum/partitions_mozarteum/excerpts/fold_4/test_photoscore.txt", 
                       base_folder="Data/Mozarteum/mozarteum_dataset/", 
                       augment=False, 
                       tokenization_mode="bekern", reduce_ratio=1.0)
    
    loader_test = DataLoader(data, batch_size=1, num_workers=20, collate_fn=batch_preparation_img2seq)
    
    model = SMTPP_Trainer.load_from_checkpoint('weights/finetuning/SMTPP_Mozarteum_Synthetic.ckpt').model

    data.set_dictionaries(model.w2i, model.i2w)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('fpw2i_not_gw2i.txt', 'r') as f:
        forbidden_tokens = f.read().split('\n')
    
    model.eval()
    ser = []
    with torch.no_grad():
        for batch in loader_test:
            x, _, y = batch
            prediction, _ = model.predict(x.to(device))
            ground_truth = [token.item() for token in y.squeeze(0)[:-1]]

            prd = []
            gtr = []
            for idx, token in enumerate(ground_truth):
                if token not in forbidden_tokens:
                    if idx < len(prediction):
                        prd.append(prediction[idx])
                    gtr.append(model.i2w[token])

            dec = "".join(prd)
            dec = dec.replace("<t>", "\t")
            dec = dec.replace("<b>", "\n")
            dec = dec.replace("<s>", " ")

            gt = "".join(gtr)
            gt = gt.replace("<t>", "\t")
            gt = gt.replace("<b>", "\n")
            gt = gt.replace("<s>", " ")
            
            ser.append(compute_poliphony_metrics([dec], [gt])[1])
    
    with open('ser_results.txt', 'w') as f:
        f.write("\n".join([str(tok) for tok in ser]))
    
if __name__ == "__main__":
    main()