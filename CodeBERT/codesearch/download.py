import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForMaskedLM

def main():
    print("loading tokenizer")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    print("loading model")
    model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained("./models/MLM_mlm/")
    tokenizer.save_pretrained("./models/MLM_mlm/")

if __name__ == '__main__':
    main()