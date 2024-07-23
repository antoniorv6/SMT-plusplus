import os
import fire
from smt_model.modeling_smt import SMTModelForCausalLM
from dotenv import load_dotenv

def push_model_to_hf(weights_path):
    print(weights_path)
    load_dotenv()
    model = SMTModelForCausalLM.from_pretrained(weights_path, use_safetensors=True, variant="Mozarteum_BeKern_fold0")
    model.push_to_hub("smt_plusplus", commit_message="mozarteum_weights_uploaded", token=os.getenv("HF_API_TOKEN"))

if __name__ == "__main__":
    fire.Fire(push_model_to_hf)