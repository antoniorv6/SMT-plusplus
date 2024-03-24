import cv2
import fire
import torch
import numpy as np
from rich import progress
from SMT import SMT
from visualizer.WeightsVisualizer import SMTWeightsVisualizer
from data_augmentation.data_augmentation import convert_img_to_tensor

def main():

    img_path = "#2.jpg"
    model_weights = "weights/fp_polish_simple/SMT_NexT_fold_0.ckpt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualizer_module = SMTWeightsVisualizer(frames_path="frames/", animation_path="animation/")
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[1] * 0.3)))
    img_tensor = convert_img_to_tensor(img)
    
    model = SMT.load_from_checkpoint(model_weights)
    w2i, i2w = model.w2i, model.i2w
    
    cache = None
    text_sequence = []
    global_weights = []
    self_weights = []
    encoder_output = model.forward_encoder(img_tensor.unsqueeze(0).to(device))
    predicted_sequence = torch.from_numpy(np.asarray([w2i['<bos>']])).to(device).unsqueeze(0)
    
    with torch.no_grad():
        for i in progress.track(range(model.maxlen)):
            output, predictions, cache, weights = model.forward_decoder(encoder_output, predicted_sequence.long(), cache=cache)
            predicted_token = torch.argmax(predictions[:, :, -1]).cpu().detach().item()
            predicted_sequence = torch.cat([predicted_sequence, torch.argmax(predictions[:, :, -1], dim=1, keepdim=True)], dim=1)
            predicted_char = i2w[predicted_token]
            if predicted_char == '<eos>':
                break
            weights["self"] = [torch.reshape(w.unsqueeze(1).cpu(), (-1, 1, encoder_output.shape[2], encoder_output.shape[3])) for w in weights["self"]]
            weights_append = weights["self"][-1]
            self_weights.append(weights["mix"][-1])
            global_weights.append(weights_append[i])
            text_sequence.append(predicted_char)
    
    attention_weights = global_weights
    attention_weights = np.stack(attention_weights, axis=0)
    attention_weights = torch.tensor(attention_weights).squeeze(1).detach().numpy()
    zero_weights = np.zeros((1, attention_weights.shape[1], attention_weights.shape[2]))
    attention_weights = np.concatenate([zero_weights, attention_weights, zero_weights], axis=0)
    
    with open("prediction.krn", "w") as predfile:
        text_sequence = "**kern \t **kern \n" + "".join(text_sequence).replace("<t>", "\t")
        text_sequence = text_sequence.replace("<b>", "\n")
        text_sequence = text_sequence.replace("<s>", " ").replace('**ekern_1.0', '**kern')
        predfile.write(text_sequence)
    
    visualizer_module.render(x=img, y="", predicted_seq=text_sequence, self_weights=self_weights, attn_weights=attention_weights, animation_name=str(0))
    
    import sys
    sys.exit()

if __name__ == "__main__":
    main()