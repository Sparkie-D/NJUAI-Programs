'''
直接在模型中使用可训练的bert
'''
import torch
import torch.nn as nn
from load_data import load_dataset
from transformers import DebertaModel, DebertaTokenizer, RobertaTokenizer, RobertaModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


class SentimentClassifier(nn.Module):
    def __init__(self, model, tokenizer, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = model
        self.bert.requires_grad_()
        self.tokenizer = tokenizer
        # self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.to(device)

    def forward(self, sentence, target_word):
        encoded_inputs = self.tokenizer(sentence, target_word, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        # 应用softmax函数获取分类概率
        return nn.functional.softmax(logits, dim=1)


def train(model, texts, targets, labels):
    print("training model...")
    model.bert.train()
    batch_size = 10
    max_epochs = 100
    loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(max_epochs):
        start = 0
        epoch_loss = 0
        while True:
            if start >= len(texts):
                break
            batch_text = texts[start:start+batch_size]
            batch_target = targets[start:start+batch_size]
            batch_label = torch.tensor(labels[start:start+batch_size], dtype=torch.long).to(device)
            batch_output = model(batch_text, batch_target)

            loss = loss_fn(batch_output, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += batch_size
            epoch_loss += loss

        if epoch % 1 == 0:
            print(f"Epoch {epoch}: loss={epoch_loss.item()} ")
            torch.save(model, f'models/roberta_model{epoch}.pth')


def predict(model, test_X):
    print("Predicting... ")
    model.bert.eval()
    texts = [item[0] for item in test_X]
    target_words = [item[1] for item in test_X]
    for i in range(len(texts)):
        texts[i].replace("$T$", target_words[i])
    predicts = []
    for text, target_word in zip(texts, target_words):
        output = model([text], [target_word])
        predicts.append(torch.argmax(output, dim=1))
    return torch.cat(predicts, dim=0)

if __name__ == "__main__":
    # 加载预训练的BERT模型和tokenizer
    model = RobertaModel.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    input_ids = tokenizer()

    # 初始化分类器
    num_classes = 3  # 情感分类的类别数
    classifier = SentimentClassifier(model, tokenizer, num_classes)

    # 加载数据
    train_set, train_label, test_set = load_dataset()
    texts = [item[0] for item in train_set]
    target_words = [item[1] for item in train_set]
    for i in range(len(texts)):
        texts[i].replace("$T$", target_words[i])

    classifier = torch.load('roberta_model40.pth').to(device) # roberta85, 0.854  #roberta40 0.8589
    # train(classifier, texts, target_words, train_label) # 再次训练

    preds = predict(classifier, test_set)
    path = '201300096.txt'
    print("Writing results into", path)
    with open(path, 'w') as f:
        for item in preds:
            val = item.item()
            f.write(str(val-1))
            f.write('\n')
