import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from core.dependencies.globals import configer, logger

class TfidfLRModel:
    """TF-IDF + Logistic Regression 模型"""
    
    def __init__(self, ngram_range=None, n_components=None, C=None, n_jobs=None):
        # 使用配置文件中的参数，如果没有传入则使用默认值
        tfidf_config = configer.tfidf_lr
        self.ngram_range = ngram_range or tuple(tfidf_config.ngram_range)
        self.n_components = n_components or tfidf_config.n_components
        self.C = C or tfidf_config.C
        self.n_jobs = n_jobs or tfidf_config.n_jobs
        self.tfidf = None
        self.svd = None
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
    
    def extract_features(self, df):
        """特征提取"""
        logger.info("提取TF-IDF特征...")
        self.tfidf = TfidfVectorizer(ngram_range=self.ngram_range)
        tfidf_feature = self.tfidf.fit_transform(df['text'])
        
        logger.info("SVD降维...")
        self.svd = TruncatedSVD(n_components=self.n_components)
        svd_feature = self.svd.fit_transform(tfidf_feature)
        
        return svd_feature
    
    def train(self, nfold=None, random_state=None):
        """训练模型"""
        # 使用配置文件中的训练参数
        training_config = configer.training
        nfold = nfold or training_config.n_folds
        random_state = random_state or training_config.random_state
        
        logger.info("="*50)
        logger.info("训练 TF-IDF + LR 模型")
        logger.info("="*50)
        
        # 加载数据
        df_train, df_test, label, df = self.load_data()
        
        # 特征提取
        features = self.extract_features(df)
        train_features = features[:-len(df_test)]
        test_features = features[-len(df_test):]

        scores = []
        kf = StratifiedKFold(n_splits=nfold, shuffle=training_config.shuffle, random_state=random_state)

        lr_oof = np.zeros((len(df_train), 2))
        lr_predictions = np.zeros((len(df_test), 2))

        for i, (train_index, valid_index) in enumerate(tqdm(kf.split(train_features, label), total=nfold, desc="TF-IDF+LR训练")):
            logger.info(f"Fold {i + 1}")
            X_train, label_train = train_features[train_index], label[train_index]
            X_valid, label_valid = train_features[valid_index], label[valid_index]

            self.model = LogisticRegression(C=self.C, n_jobs=self.n_jobs)
            self.model.fit(X_train, label_train)

            lr_oof[valid_index] = self.model.predict_proba(X_valid)
            scores.append(roc_auc_score(label_valid, lr_oof[valid_index][:,1]))

            lr_predictions += self.model.predict_proba(test_features) / nfold
            logger.info(f"当前fold AUC: {scores[-1]:.4f}")

        logger.info(f"TF-IDF + LR 平均 AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return np.mean(scores), lr_predictions[:,1]
    
    def save_results(self, predictions):
        """保存结果"""
        import os
        results_dir = configer.training.results_dir
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "tfidf_lr_result.csv")
        pd.DataFrame(predictions).to_csv(result_file, index=False, header=False)
        logger.info(f"结果已保存到 {result_file}")

if __name__ == "__main__":
    model = TfidfLRModel()
    auc, predictions = model.train()
    model.save_results(predictions) 