import cv2
import fire
import torch
import numpy as np
from rich import progress
from ModelManager import SMT
from data_augmentation.data_augmentation import convert_img_to_tensor

WIDTH = int(2100 * 0.5)

def main(img_path, model_weights):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (WIDTH, int(img.shape[0] * WIDTH / img.shape[1])))
    img = convert_img_to_tensor(img)
    
    model = SMT.load_from_checkpoint(model_weights)
    w2i, i2w = model.w2i, model.i2w
    
    cache = None
    text_sequence = []
    encoder_output = model.forward_encoder(img.unsqueeze(0).to(device))
    predicted_sequence = torch.from_numpy(np.asarray([w2i['<bos>']])).to(device).unsqueeze(0)
    
    with torch.no_grad():
        for i in progress.track(range(model.maxlen)):
            output, predictions, cache, weights = model.forward_decoder(encoder_output, predicted_sequence.long(), cache=cache)
            predicted_token = torch.argmax(predictions[:, :, -1]).cpu().detach().item()
            predicted_sequence = torch.cat([predicted_sequence, torch.argmax(predictions[:, :, -1], dim=1, keepdim=True)], dim=1)
            predicted_char = i2w[predicted_token]
            if predicted_char == '<eos>':
                break
            text_sequence.append(predicted_char)
    
    with open("prediction.krn", "w") as predfile:
        text_sequence = "**kern \t **kern \n" + "".join(text_sequence).replace("<t>", "\t")
        text_sequence = text_sequence.replace("<b>", "\n")
        text_sequence = text_sequence.replace("<s>", " ").replace('**ekern_1.0', '**kern')
        predfile.write(text_sequence)
    
    import sys
    sys.exit()
    
    
    pass

def launch(img_path, model_weights):
    main(img_path, model_weights)

if __name__ == "__main__":
    fire.Fire(launch)