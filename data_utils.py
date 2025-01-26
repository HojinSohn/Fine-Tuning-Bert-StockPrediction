from transformers import BertTokenizer
import torch
import pandas as pd
import ast


def getTokenizer(model_name):
    return BertTokenizer.from_pretrained(model_name)

# Tokenize the text data
def tokenize_function(text, model_name):
    # Initialize the tokenizer
    tokenizer = getTokenizer(model_name)
    return tokenizer(text, padding='max_length', truncation=True, max_length=300, return_tensors='pt') #length

def getDataFrame():
    data_csv = "./data_files/train_data.csv"
    df = pd.read_csv(data_csv)
    return df
    
def get_tokenized_data(model_name):
    df = getDataFrame()
    
    tokenized_texts = df['text'].apply(lambda x: tokenize_function(x, model_name)) #bert-base-uncased'

    # Prepare input tensors for the model
    input_ids = torch.cat([x['input_ids'] for x in tokenized_texts], dim=0)
    attention_masks = torch.cat([x['attention_mask'] for x in tokenized_texts], dim=0)

    
    df['labels'] = df['Open_Close_diff'].apply(lambda x: 1 if x > 0 else 0)
    
    # Prepare labels (stock price change)
    labels = torch.tensor(df['labels'].values)
    
    return [tokenized_texts, input_ids, attention_masks, labels]
