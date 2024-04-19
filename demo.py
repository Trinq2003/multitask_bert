import torch
from multitask_classifier import MultitaskBERT
from demo.options import demo_get_args
from demo.demo_datasets import load_demo_multitask_data, \
SentenceClassificationTestDataset, SentencePairTestDataset
import numpy as np


if __name__ == '__main__':
    args = demo_get_args()
    sentence = "The film is really amazing"
    sent1 = "What is Amartya Sen famous for?"
    sent2 = "Why is Amartya Sen famous?"

    device = torch.device('mps') if args.use_gpu else torch.device('cpu')
    saved = torch.load("weights/finetune-10-1e-05-multitask.pt", map_location='cpu')
    config = saved['model_config']

    model = MultitaskBERT(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)

    
    if args.sentiment_anal:
        demo = load_demo_multitask_data(args, sentence)
        demo = SentenceClassificationTestDataset(demo)
        token_ids = demo.collate_fn(demo.dataset)['token_ids']
        attention_mask = demo.collate_fn(demo.dataset)['attention_mask']

        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)

        print(sentence)
        logits = model.predict_sentiment(token_ids, attention_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        print("Score: ", preds)
    else:
        demo = load_demo_multitask_data(args, sent1, sent2)
        demo = SentencePairTestDataset(demo)
        token_ids_1 = demo.collate_fn(demo.dataset)['token_ids_1']
        attention_mask_1 = demo.collate_fn(demo.dataset)['attention_mask_1']
        token_ids_2 = demo.collate_fn(demo.dataset)['token_ids_2']
        attention_mask_2 = demo.collate_fn(demo.dataset)['attention_mask_2']

        token_ids_1 = token_ids_1.to(device)
        attention_mask_1 = attention_mask_1.to(device)
        token_ids_2 = token_ids_2.to(device)
        attention_mask_2 = attention_mask_2.to(device)

        print(sent1)
        print(sent2)

        if args.paraphrase_detect:
            logits = model.predict_para(model.sent_pair_linear(token_ids_1, attention_mask_1, token_ids_2, attention_mask_2, device))
            print(logits)
            y_hat = logits.sigmoid().round().flatten().detach().cpu().numpy()
            print("Score: ", y_hat)
        
        elif args.semantic_sim:
            logits = model.predict_sim(model.sent_pair_linear(token_ids_1, attention_mask_1, token_ids_2, attention_mask_2, device))
            print(logits)
            y_hat = logits.flatten().detach().cpu().numpy()
            print("Score: ", y_hat)
    
    
    
    