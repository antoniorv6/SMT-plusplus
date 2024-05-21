import cv2
import torch
import numpy as np
from rich import progress
import hydra
from SMT import SMT
from visualizer.WeightsVisualizer import SMTWeightsVisualizer
from data_augmentation.data_augmentation import convert_img_to_tensor
from config_typings import Config
from data import GraphicCLDataModule
from eval_functions import compute_poliphony_metrics
import os

@hydra.main(version_base=None, config_path="config")
def main(config:Config):

    data_module = GraphicCLDataModule(config.data, config.cl, fold=config.data.fold)
    test_set = data_module.test_dataloader()
    test_dataset = data_module.test_dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SMT.load_from_checkpoint("weights/fp_grandstaff/SMT_NexT_ekern_fold_0.ckpt")
    w2i, i2w = model.w2i, model.i2w

    ser_values = []
    os.makedirs("predictions_by_SER", exist_ok=True)
    
    sample_index = 0
    for (sample, _, gt) in test_set:
        gt = [test_dataset.i2w[token] for token in gt[0].detach().numpy() if test_dataset.i2w[token] != '<pad>']
        cache = None
        text_sequence = []
        global_weights = []
        self_weights = []
        encoder_output = model.forward_encoder(sample.to(device))
        predicted_sequence = torch.from_numpy(np.asarray([w2i['<bos>']])).to(device).unsqueeze(0)

        with torch.no_grad():
            for i in range(model.maxlen):
                output, predictions, cache, weights = model.forward_decoder(encoder_output, predicted_sequence.long(), cache=cache)
                predicted_token = torch.argmax(predictions[:, :, -1]).cpu().detach().item()
                predicted_sequence = torch.cat([predicted_sequence, torch.argmax(predictions[:, :, -1], dim=1, keepdim=True)], dim=1)
                predicted_char = i2w[predicted_token]
                if predicted_char == '<eos>':
                    break
                text_sequence.append(predicted_char)
        
        dec = "".join(text_sequence)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")
        #print(dec)

        gt = "".join(gt)
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")
        #print(gt)
        
        _, ser, _ = compute_poliphony_metrics([dec], [gt])

        
        #round ser to the unit
        ser = round(ser)

        if ser > 100:
            ser = 100

        if not os.path.exists("predictions_by_SER/" + str(ser)):
            os.makedirs("predictions_by_SER/" + str(ser), exist_ok=True)
        
        with open("predictions_by_SER/" + str(ser) + f"/prediction_{sample_index}.krn", "w") as predfile:
            predfile.write(dec)
        
        with open("predictions_by_SER/" + str(ser) + f"/gt_{sample_index}.krn", "w") as gtfile:
            gtfile.write(gt)
        
        #Put the image also
        img = sample[0].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img * 255
        img = img.astype(np.uint8)
        cv2.imwrite("predictions_by_SER/" + str(ser) + f"/sample_{sample_index}.png", img)
       
        ser_values.append(ser)

        sample_index += 1
    
    #Plot a histogram with the SER values
    import matplotlib.pyplot as plt
    plt.hist(ser_values, bins=20)
    #save the image as ser_hystogram.png
    plt.savefig("ser_hystogram.png")
            
    #attention_weights = global_weights
    #attention_weights = np.stack(attention_weights, axis=0)
    #attention_weights = torch.tensor(attention_weights).squeeze(1).detach().numpy()
    #zero_weights = np.zeros((1, attention_weights.shape[1], attention_weights.shape[2]))
    #attention_weights = np.concatenate([zero_weights, attention_weights, zero_weights], axis=0)
    #
    #with open("prediction.krn", "w") as predfile:
    #    text_sequence = "**kern \t **kern \n" + "".join(text_sequence).replace("<t>", "\t")
    #    text_sequence = text_sequence.replace("<b>", "\n")
    #    text_sequence = text_sequence.replace("<s>", " ").replace('**ekern_1.0', '**kern')
    #    predfile.write(text_sequence)
    
    #visualizer_module.render(x=img, y="", predicted_seq=pred_sequence, self_weights=self_weights, attn_weights=attention_weights, animation_name=str(0))
    
    print("SER values: ", ser_values)
    print("Mean SER: ", np.mean(ser_values))
    print("Median SER: ", np.median(ser_values))

if __name__ == "__main__":
    main()