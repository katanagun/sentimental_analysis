import joblib
from threading import Thread
import time

# Загрузка модели и векторизатора
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Пример данных для обработки
reviews_to_process = [
    'Perfect for me! Exactly what I needed and at such a great price.',
    'Item as described. Happy with purchase.',
    'This product exceeded my expectations for both quality and value.',
    'Customer service was excellent, the entire process was smooth and efficient.',
    'This product provides excellent performance and durability. We highly recommend it.',
    'These candles are so cheap looking, the Dollar Store would have second hand embarrassment. The only use I can see for these candles is inserting them into another object that needs some lighting. Spend the extra money and get some candles with the wax coating. I should have known better since I saw an Amazon comment about this product having a high return rate.',
    'The first set I got only one candle out of 12 worked. So I asked for a new one seller was quick with sending a new one that also did not work. I bought two didn’t brands of batteries thinking maybe it was the issue but nope. Returned both sets. Don’t buy.',
    'This product is a fantastic product! I’ve been using it now, and it hasn’t disappointed. The quality is outstanding, and it’s incredibly durable. The value for the price is excellent. Shipping was fast, and the packaging was very well-protected. I’m highly satisfied with my purchase. Highly recommend!',
    'I’m absolutely thrilled with this product! From the moment I unboxed it, I was impressed by the quality and craftsmanship. The design is sleek and modern, and it feels incredibly sturdy.'
]

# Функция для предсказания сентимента
def predict_sentiment(review):
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)
    print(f'Review: {review}\nSentiment: {prediction[0]}\n')

# Запуск программы и отслеживание времени выполнения
start_time = time.time()

# Создание и запуск потоков
threads = []
for review in reviews_to_process:
    thread = Thread(target=predict_sentiment, args=(review,))
    threads.append(thread)
    thread.start()

# Ожидание завершения всех потоков
for thread in threads:
    thread.join()

end_time = time.time()
execution_time = end_time - start_time

print(f'Общее время выполнения программы: {execution_time:.4f} секунд')