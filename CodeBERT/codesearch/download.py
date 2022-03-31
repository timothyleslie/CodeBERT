import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForMaskedLM, T5ForConditionalGeneration, RobertaForSequenceClassification

def main():
    # print("loading tokenizer")
    # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    # print("loading model")
    # model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    # model_to_save = model.module if hasattr(model,
    #                                             'module') else model  # Take care of distributed/parallel training
    # model_to_save.save_pretrained("./models/MLM_mlm/")
    # tokenizer.save_pretrained("./models/MLM_mlm/")


    # print("loading tokenizer")
    # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    # print("loading model")
    # model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base")
    # model_to_save = model.module if hasattr(model,
    #                                             'module') else model  # Take care of distributed/parallel training
    # model_to_save.save_pretrained("./models/MLM_base/")
    # tokenizer.save_pretrained("./models/MLM_base/")


    # print("loading tokenizer")
    # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    # print("loading model")
    # model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base")
    # model_to_save = model.module if hasattr(model,
    #                                             'module') else model  # Take care of distributed/parallel training
    # model_to_save.save_pretrained("./models/Finetune/")
    # tokenizer.save_pretrained("./models/Finetune/")


    print("loading tokenizer")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    print("loading model")
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained("./models/T5_base/")
    tokenizer.save_pretrained("./models/T5_base/")


    print("loading tokenizer")
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    print("loading model")
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
    model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained("./models/T5_small/")
    tokenizer.save_pretrained("./models/T5_small/")

if __name__ == '__main__':
    main()