import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Пример данных (замените на ваши реальные данные)
positive_reviews = [
    'This product is absolutely amazing! It really exceeded my expectations.',
    'Love it! So much better than I expected. Highly recommend!',
    'I absolutely love it! The quality is incredible; five stars!',
    'Fast shipping and good packaging.',
    'Amazing customer service.',
    'Great product, highly recommend.',
    'Five stars, very happy!',
    'Will definitely order again.',
    'Top-notch quality and service.',
    'Best purchase I have made online.',
    'Very pleased with the product.',
    'Fantastic experience overall.',
    'Highly satisfied with the quality.',
    'Great value for the price.',
    'Product works perfectly.',
    'Impressed with the durability.',
    'Outstanding performance.',
    'Love this product!',
    'Exceeded my expectations.',
    'Very happy with my purchase.',
    'Great quality and fast delivery.',
    'Superb customer service.',
    'Product is as described.',
    'Will buy from this seller again.',
    'Very good quality product.',
    'Happy with the purchase.',
    'Excellent value for money.',
    'Product arrived on time.',
    'Very reliable seller.',
    'Highly recommend this product.',
    'Product is worth every penny.',
    'Very satisfied with the service.',
    'Great experience overall.',
    'Product is of high quality.',
    'Very pleased with the purchase.',
    'Fantastic product and service.',
    'Will definitely recommend to others.',
    'Product is exactly what I needed.',
    'Very happy with the quality.',
    'Absolutely love this product!'
]

negative_reviews = [
    'Terrible product, do not buy!',
    'Poor quality, very disappointed.',
    'Not satisfied with my purchase.',
    'Slow shipping and bad packaging.',
    'Horrible customer service.',
    'Product did not meet my expectations.',
    'One star, very unhappy!',
    'Will never order again.',
    'Low-quality product and service.',
    'Worst purchase I have made online.',
    'Very disappointed with the product.',
    'Awful experience overall.',
    'Not satisfied with the quality.',
    'Poor value for the price.',
    'Product does not work properly.',
    'Disappointed with the durability.',
    'Terrible performance.',
    'Hate this product!',
    'Did not meet my expectations.',
    'Very unhappy with my purchase.',
    'Poor quality and slow delivery.',
    'Terrible customer service.',
    'Product is not as described.',
    'Will not buy from this seller again.',
    'Very poor quality product.',
    'Unhappy with the purchase.',
    'Not worth the money spent.',
    'Product arrived late.',
    'Unreliable seller.',
    'Do not recommend this product.',
    'Product is a waste of money.',
    'Not satisfied with the service.',
    'Bad experience overall.',
    'Product is of low quality.',
    'Very disappointed with the purchase.',
    'Terrible product and service.',
    'Will not recommend to others.',
    'Product is not what I needed.',
    'Very unhappy with the quality.',
]

# Создание DataFrame
data = pd.DataFrame({
    'review': positive_reviews + negative_reviews,
    'sentiment': ['positive'] * len(positive_reviews) + ['negative'] * len(negative_reviews)
})

# Предобработка данных
X = data['review']
y = data['sentiment']

# Векторизация текстовых данных с использованием TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_tfidf, y)

# Save the model
joblib.dump(model, 'sentiment_model.pkl')

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')