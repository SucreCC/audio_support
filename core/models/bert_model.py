import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW

from core.dependencies.globals import configer, logger

# 设置设备
device = configer.system.get_device()

class BertDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class BertClassifier(nn.Module):
    def __init__(self, bert_model_name=None, num_classes=None):
        super(BertClassifier, self).__init__()
        bert_config = configer.bert
        bert_model_name = bert_model_name or bert_config.model_name
        num_classes = num_classes or bert_config.num_classes
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(bert_config.dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class BertModelWrapper:
    """BERT模型包装器"""
    
    def __init__(self, bert_model_name=None, max_len=None, 
                 batch_size=None, epochs=None, lr=None):
        # 使用配置文件中的参数
        bert_config = configer.bert
        self.bert_model_name = bert_model_name or bert_config.model_name
        self.max_len = max_len or bert_config.max_len
        self.batch_size = batch_size or bert_config.batch_size
        self.epochs = epochs or bert_config.epochs
        self.lr = lr or bert_config.learning_rate
        self.tokenizer = None
        self.model = None
        
    def load_data(self):
        """加载数据"""
        logger.info("加载数据...")
        data_config = configer.data
        df_train = pd.read_table(data_config.train_file, 
                                 names=[data_config.columns['q1'], data_config.columns['q2'], data_config.columns['label']]).fillna(data_config.fill_na)
        df_test = pd.read_table(data_config.test_file, 
                                names=[data_config.columns['q1'], data_config.columns['q2']]).fillna(data_config.fill_na)
        label = df_train[data_config.columns['label']].values
        df = pd.concat([df_train, df_test], ignore_index=True)
        df['text'] = df[data_config.columns['q1']] + data_config.text_separator + df[data_config.columns['q2']]
        
        logger.info(f"训练集大小: {len(df_train)}")
        logger.info(f"测试集大小: {len(df_test)}")
        
        return df_train, df_test, label, df
    
    def init_tokenizer(self):
        """初始化tokenizer"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
            logger.info("使用中文BERT模型")
        except:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            logger.info("使用英文BERT模型")
    
    def train_bert_model(self, model, train_loader, val_loader):
        """BERT训练函数"""
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        best_val_auc = 0
        val_predictions = []
        
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 验证
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels']
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs, dim=1)
                    val_preds.extend(probs[:, 1].cpu().numpy())
                    val_labels.extend(labels.numpy())
            
            val_auc = roc_auc_score(val_labels, val_preds)
            logger.info(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                val_predictions = val_preds
        
        return best_val_auc, val_predictions

    def predict_bert(self, model, test_loader):
        """BERT预测函数"""
        model.eval()
        test_predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)
                test_predictions.extend(probs[:, 1].cpu().numpy())
        
        return test_predictions
    
    def train(self, nfold=None, random_state=None):
        """训练模型"""
        # 使用配置文件中的训练参数
        training_config = configer.training
        nfold = nfold or training_config.n_folds
        random_state = random_state or training_config.random_state
        
        logger.info("="*50)
        logger.info("训练 BERT 微调模型")
        logger.info("="*50)
        
        # 加载数据
        df_train, df_test, label, df = self.load_data()
        
        # 初始化tokenizer
        self.init_tokenizer()

        scores = []
        kf = StratifiedKFold(n_splits=nfold, shuffle=training_config.shuffle, random_state=random_state)
        bert_predictions = np.zeros((len(df_test), 2))

        for i, (train_index, valid_index) in enumerate(tqdm(kf.split(df_train, label), total=nfold, desc="BERT训练")):
            logger.info(f"BERT Fold {i + 1}")
            
            # 准备数据
            train_texts = df_train.iloc[train_index]['text'].values
            val_texts = df_train.iloc[valid_index]['text'].values
            test_texts = df_test['text'].values
            
            train_labels = label[train_index]
            val_labels = label[valid_index]
            
            # 创建数据集
            train_dataset = BertDataset(train_texts, train_labels, self.tokenizer, self.max_len)
            val_dataset = BertDataset(val_texts, val_labels, self.tokenizer, self.max_len)
            test_dataset = BertDataset(test_texts, None, self.tokenizer, self.max_len)
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # 训练模型
            model = BertClassifier(self.bert_model_name)
            val_auc, val_preds = self.train_bert_model(model, train_loader, val_loader)
            
            scores.append(val_auc)
            
            # 预测测试集
            test_preds = self.predict_bert(model, test_loader)
            bert_predictions[:, 1] += np.array(test_preds) / nfold
            bert_predictions[:, 0] += (1 - np.array(test_preds)) / nfold
            
            logger.info(f"BERT Fold {i+1} AUC: {val_auc:.4f}")

        logger.info(f"BERT 平均 AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return np.mean(scores), bert_predictions[:,1]
    
    def save_results(self, predictions):
        """保存结果"""
        import os
        results_dir = configer.training.results_dir
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "bert_result.csv")
        pd.DataFrame(predictions).to_csv(result_file, index=False, header=False)
        logger.info(f"结果已保存到 {result_file}")

if __name__ == "__main__":
    model = BertModelWrapper()
    auc, predictions = model.train()
    model.save_results(predictions) 