import os
import re
import pickle
import nltk
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Thêm dòng này
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)  # Thêm dòng này để cho phép CORS

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
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Huấn luyện mô hình Naive Bayes
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Lưu mô hình và vectorizer để sử dụng sau này
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comment = data['comment']
    
    # Làm sạch và vector hóa bình luận
    clean_comment = clean_text(comment)
    comment_vect = vectorizer.transform([clean_comment])
    
    # Dự đoán
    prediction = model.predict(comment_vect)
    
    # Trả về kết quả
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    return jsonify({'comment': comment, 'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
