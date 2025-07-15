import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
from gensim.models import Word2Vec

from core.dependencies.globals import configer, logger

# 设置设备
device = configer.system.get_device()

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=None, hidden_dim=None, num_layers=None, num_classes=None):
        super(LSTMClassifier, self).__init__()
        lstm_config = configer.lstm
        embedding_dim = embedding_dim or lstm_config.embedding_dim
        hidden_dim = hidden_dim or lstm_config.hidden_dim
        num_layers = num_layers or lstm_config.num_layers
        num_classes = num_classes or lstm_config.num_classes
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=lstm_config.dropout, bidirectional=lstm_config.bidirectional)
        self.dropout = nn.Dropout(lstm_config.dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.classifier(lstm_out)
        return output

class Word2VecDataset(Dataset):
    def __init__(self, texts, labels=None, w2v_model=None, max_len=50):
        self.texts = texts
        self.labels = labels
        self.w2v_model = w2v_model
        self.max_len = max_len
        
        # 创建词汇表
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for word in w2v_model.wv.index_to_key:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.texts[idx]
        # 将token转换为索引
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # 截断或填充到固定长度
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.word2idx['<PAD>']] * (self.max_len - len(indices))
        
        item = {'input_ids': torch.tensor(indices, dtype=torch.long)}
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class LstmModelWrapper:
    """Word2Vec + LSTM模型包装器"""
    
    def __init__(self, embedding_dim=None, hidden_dim=None, num_layers=None, 
                 max_len=None, batch_size=None, epochs=None, lr=None,
                 vector_size=None, window=None, min_count=None, workers=None, sg=None):
        # 使用配置文件中的参数
        lstm_config = configer.lstm
        self.embedding_dim = embedding_dim or lstm_config.embedding_dim
        self.hidden_dim = hidden_dim or lstm_config.hidden_dim
        self.num_layers = num_layers or lstm_config.num_layers
        self.max_len = max_len or lstm_config.max_len
        self.batch_size = batch_size or lstm_config.batch_size
        self.epochs = epochs or lstm_config.epochs
        self.lr = lr or lstm_config.learning_rate
        self.vector_size = vector_size or lstm_config.vector_size
        self.window = window or lstm_config.window
        self.min_count = min_count or lstm_config.min_count
        self.workers = workers or lstm_config.workers
        self.sg = sg or lstm_config.sg
        self.w2v_model = None
        
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
    
    def tokenize_data(self, df):
        """分词处理"""
        logger.info("进行中文分词...")
        def tokenize_text(text):
            return list(jieba.cut(str(text)))

        df['tokens'] = df['text'].apply(tokenize_text)
        return df
    
    def train_word2vec(self, sentences):
        """训练Word2Vec模型"""
        logger.info("训练Word2Vec模型...")
        self.w2v_model = Word2Vec(sentences, vector_size=self.vector_size, 
                                 window=self.window, min_count=self.min_count, 
                                 workers=self.workers, sg=self.sg)
        
        # 保存Word2Vec模型
        import os
        results_dir = configer.training.results_dir
        os.makedirs(results_dir, exist_ok=True)
        model_file = os.path.join(results_dir, "word2vec_model.model")
        self.w2v_model.save(model_file)
        logger.info(f"Word2Vec模型已保存到 {model_file}")
    
    def train_lstm_model(self, model, train_loader, val_loader):
        """LSTM训练函数"""
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        best_val_auc = 0
        val_predictions = []
        
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"LSTM Epoch {epoch+1}/{self.epochs}"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids)
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
                    labels = batch['labels']
                    
                    outputs = model(input_ids)
                    probs = torch.softmax(outputs, dim=1)
                    val_preds.extend(probs[:, 1].cpu().numpy())
                    val_labels.extend(labels.numpy())
            
            val_auc = roc_auc_score(val_labels, val_preds)
            logger.info(f"LSTM Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                val_predictions = val_preds
        
        return best_val_auc, val_predictions

    def predict_lstm(self, model, test_loader):
        """LSTM预测函数"""
        model.eval()
        test_predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                
                outputs = model(input_ids)
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
        logger.info("训练 Word2Vec + LSTM 模型")
        logger.info("="*50)
        
        # 加载数据
        df_train, df_test, label, df = self.load_data()
        
        # 分词处理
        df = self.tokenize_data(df)
        
        # 训练Word2Vec模型
        sentences = df['tokens'].values
        self.train_word2vec(sentences)

        scores = []
        kf = StratifiedKFold(n_splits=nfold, shuffle=training_config.shuffle, random_state=random_state)
        lstm_predictions = np.zeros((len(df_test), 2))

        for i, (train_index, valid_index) in enumerate(tqdm(kf.split(df_train, label), total=nfold, desc="LSTM训练")):
            logger.info(f"LSTM Fold {i + 1}")
            
            # 准备数据
            train_tokens = df_train.iloc[train_index]['tokens'].values
            val_tokens = df_train.iloc[valid_index]['tokens'].values
            test_tokens = df_test['tokens'].values
            
            train_labels = label[train_index]
            val_labels = label[valid_index]
            
            # 创建数据集
            train_dataset = Word2VecDataset(train_tokens, train_labels, self.w2v_model, self.max_len)
            val_dataset = Word2VecDataset(val_tokens, val_labels, self.w2v_model, self.max_len)
            test_dataset = Word2VecDataset(test_tokens, None, self.w2v_model, self.max_len)
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # 训练模型
            vocab_size = len(train_dataset.word2idx)
            model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=self.embedding_dim,
                                 hidden_dim=self.hidden_dim, num_layers=self.num_layers)
            val_auc, val_preds = self.train_lstm_model(model, train_loader, val_loader)
            
            scores.append(val_auc)
            
            # 预测测试集
            test_preds = self.predict_lstm(model, test_loader)
            lstm_predictions[:, 1] += np.array(test_preds) / nfold
            lstm_predictions[:, 0] += (1 - np.array(test_preds)) / nfold
            
            logger.info(f"LSTM Fold {i+1} AUC: {val_auc:.4f}")

        logger.info(f"Word2Vec + LSTM 平均 AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return np.mean(scores), lstm_predictions[:,1]
    
    def save_results(self, predictions):
        """保存结果"""
        import os
        results_dir = configer.training.results_dir
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "lstm_result.csv")
        pd.DataFrame(predictions).to_csv(result_file, index=False, header=False)
        logger.info(f"结果已保存到 {result_file}")

if __name__ == "__main__":
    model = LstmModelWrapper()
    auc, predictions = model.train()
    model.save_results(predictions) 