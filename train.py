from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification
from transformers import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import softmax
from torch.cuda import is_available
from sklearn.metrics import accuracy_score
import data_utils as du
import torch

def train():
#     model_name = 'yiyanghkust/finbert-tone'
    model_name = 'bert-base-uncased'
    tokenized_texts, input_ids, attention_masks, labels = du.get_tokenized_data(model_name) # change finance or what
    
    # Split into train and validation sets
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
        input_ids, attention_masks, labels, test_size=0.2, random_state=42
    )

    # Create TensorDatasets
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

    # Create DataLoaders
    batch_size = 32 

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load pretrained BERT model with a classification head
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,  # Binary classification
        output_attentions=False,
        output_hidden_states=False
    )
   

    # sentiment analsis (financial) first
    #model_finance_sentiment = BertForSequenceClassification.from_pretrained(
    #    model_name, 
    #    num_labels=2,
    #    ignore_mismatched_sizes=True,
    #    output_attentions=False,
    #    output_hidden_states=False
    #)
    
    # set model to finance / change if needed
    #model = model_finance_sentiment
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=25e-6, eps=1e-8)

    # Loss function for binary classification
    loss_fn = BCEWithLogitsLoss()
    
    #Use GPU if available
    device = torch.device("cuda" if is_available() else "cpu")
    model.to(device)
    print(device)

    # Training loop
    epochs = 4

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Zero gradients
            model.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}: Average Training Loss = {avg_train_loss}")

    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Forward pass
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_input_mask
            )
            logits = outputs.logits
            preds = torch.argmax(softmax(logits, dim=1), dim=1).cpu().numpy()
            label_ids = b_labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(label_ids)

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {accuracy}")
    
    torch.save(model.state_dict(), 'bert_stock_predictor.pth')

if __name__ == "__main__":
    train()
