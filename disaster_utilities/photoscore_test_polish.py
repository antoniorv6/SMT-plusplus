import torch
from torch.utils.data import DataLoader
from data import RealDataset, batch_preparation_img2seq
from smt_trainer import SMTPP_Trainer
from eval_functions import compute_poliphony_metrics
from utils.vocab_utils import check_and_retrieveVocabulary
import os

torch.set_float32_matmul_precision('high')

def main():
    data = RealDataset(data_path=f"Data/Polish_Scores/partitions_polish_scores/excerpts/fold_5/specific_test.txt", 
                       base_folder="Data/Polish_Scores/polish_scores_dataset/", 
                       augment=False, 
                       tokenization_mode="bekern", reduce_ratio=0.5)
    
    loader_test = DataLoader(data, batch_size=1, num_workers=20, collate_fn=batch_preparation_img2seq)
    
    model = SMTPP_Trainer.load_from_checkpoint('weights/finetuning/SMTPP_Mozarteum_Synthetic.ckpt').model
    
    realw2i, reali2w = check_and_retrieveVocabulary([], "vocab/", "Polish_Scores_BeKern")
    data.set_dictionaries(realw2i, reali2w)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    ser = []
    os.makedirs('ser_results', exist_ok=True)
    with torch.no_grad():
        for batch in loader_test:
            x, _, y = batch
            prediction, _ = model.predict(x.to(device))
            ground_truth = [token.item() for token in y.squeeze(0)[:-1]]

            #prd = [reali2w[token] for token in prediction]
            gtr = [reali2w[token] for token in ground_truth]

            dec = "".join(prediction)
            dec = dec.replace("<t>", "\t")
            dec = dec.replace("<b>", "\n")
            dec = dec.replace("<s>", " ")
            
            with open(f'ser_results/28_prediction_finetuned.ekern', 'w') as f:
                f.write(dec)

            gt = "".join(gtr)
            gt = gt.replace("<t>", "\t")
            gt = gt.replace("<b>", "\n")
            gt = gt.replace("<s>", " ")
            
            with open(f'ser_results/35_prediction_finetuned.ekern', 'w') as f:
                f.write(gt)
            
            #Save x as an image
            x = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
            x = (x * 255).astype('uint8')
            x = x[:, :, 0]
            x = 255 - x
            from PIL import Image
            img = Image.fromarray(x)
            img.save(f'ser_results/35_prediction_finetuned.png')
            
            ser.append(compute_poliphony_metrics([dec], [gt])[1])
    
    with open('ser_results.txt', 'w') as f:
        f.write("\n".join([str(tok) for tok in ser]))
    
if __name__ == "__main__":
    main()