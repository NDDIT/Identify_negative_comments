import os
import re
import pickle
import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)  # Cho phép CORS

# Tải dữ liệu IMDB
def load_imdb_data(directory):
    reviews = []
    sentiments = []
    
    for label in ['pos', 'neg']:
        dir_path = os.path.join(directory, label)
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    sentiments.append(1 if label == 'pos' else 0)  # 1: positive, 0: negative
    
    return pd.DataFrame({'review': reviews, 'sentiment': sentiments})

# Đường dẫn tới thư mục chứa dữ liệu IMDB
data_dir = './aclImdb_v1/aclImdb/train'  # Đảm bảo đường dẫn này đúng
data = load_imdb_data(data_dir)

# Xử lý dữ liệu (Data Preprocessing)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Loại bỏ HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Loại bỏ ký tự đặc biệt
    text = text.lower()  # Chuyển thành chữ thường
    text = text.split()  # Tokenize
    text = [word for word in text if word not in stop_words]  # Loại bỏ stop words
    return ' '.join(text)

# Áp dụng hàm xử lý văn bản cho cột 'review'
data['clean_review'] = data['review'].apply(clean_text)

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data['clean_review'], data['sentiment'], test_size=0.2, random_state=42)

# Vector hóa văn bản thành các vector số (Bag of Words)
vectorizer = CountVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train).toarray()
X_test_vect = vectorizer.transform(X_test).toarray()

# Tạo lớp MultinomialNB từ đầu
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.classes_ = []
    
    def fit(self, X, y, epochs=10):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        
        # Tính xác suất P(class)
        class_counts = np.bincount(y)
        self.class_log_prior_ = {c: np.log(class_counts[i] / n_samples) for i, c in enumerate(self.classes_)}
        
        # Tính xác suất P(word|class)
        word_count = defaultdict(lambda: np.zeros(n_features))
        total_count = defaultdict(int)
        
        accuracy_list = []
        loss_list = []
        
        for epoch in range(epochs):
            for i, label in enumerate(y):
                word_count[label] += X[i]
                total_count[label] += X[i].sum()
            
            # Tính log xác suất với Laplace smoothing
            for c in self.classes_:
                self.feature_log_prob_[c] = np.log((word_count[c] + self.alpha) / (total_count[c] + self.alpha * n_features))
            
            # Dự đoán trên tập huấn luyện
            y_pred_train = self.predict(X)
            
            # Tính accuracy
            acc = accuracy_score(y, y_pred_train)
            accuracy_list.append(acc)
            
            # Tính loss (giả sử loss là 1 - accuracy cho ví dụ này)
            loss = 1 - acc
            loss_list.append(loss)
            
            print(f'Epoch {epoch + 1}/{epochs} - Accuracy: {acc:.4f}, Loss: {loss:.4f}')

        return accuracy_list, loss_list

    def predict(self, X):
        results = []
        for x in X:
            log_probs = {}
            for c in self.classes_:
                log_prob = self.class_log_prior_[c] + x @ self.feature_log_prob_[c]
                log_probs[c] = log_prob
            results.append(max(log_probs, key=log_probs.get))
        return np.array(results)

# Khởi tạo và huấn luyện mô hình
nb_model = MultinomialNB()
epochs = 10
accuracy_list, loss_list = nb_model.fit(X_train_vect, y_train, epochs)

# Lưu mô hình và vectorizer để sử dụng sau này
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Dự đoán trên tập kiểm tra
y_pred = nb_model.predict(X_test_vect)

# Tính toán độ chính xác, precision, recall và F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Vẽ biểu đồ cho độ chính xác và tổn thất
plt.figure(figsize=(12, 6))

# Vẽ Accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), accuracy_list, marker='o', color='blue', label='Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(range(1, epochs + 1))
plt.grid()
plt.legend()

# Vẽ Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), loss_list, marker='o', color='red', label='Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.xticks(range(1, epochs + 1))
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# API dự đoán cảm xúc
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comment = data['comment']
    
    # Làm sạch và vector hóa bình luận
    clean_comment = clean_text(comment)
    comment_vect = vectorizer.transform([clean_comment]).toarray()
    
    # Dự đoán
    prediction = nb_model.predict(comment_vect)
    
    # Trả về kết quả
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    return jsonify({'comment': comment, 'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
