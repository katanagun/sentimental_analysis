import joblib
import time

# Загрузка модели и векторизатора
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Пример данных для обработки
reviews_to_process = [
    'We are highly satisfied with this product. The quality is excellent and the service was prompt.',
    'The product met our needs perfectly. Excellent value for money.',
    'This product offers superior performance and durability. We wholeheartedly recommend it.',
    'Exceeded my expectations. Durable and functional.',
    'This is a total winner! Fast shipping, great packaging, and amazing customer service.',
    'These candles are so cheap looking, the Dollar Store would have second hand embarrassment. The only use I can see for these candles is inserting them into another object that needs some lighting. Spend the extra money and get some candles with the wax coating. I should have known better since I saw an Amazon comment about this product having a high return rate.',
    'The first set I got only one candle out of 12 worked. So I asked for a new one seller was quick with sending a new one that also did not work. I bought two didn’t brands of batteries thinking maybe it was the issue but nope. Returned both sets. Don’t buy.',
    'The only good thing about these are that they aren’t wax. This is only a plastic looking Hershey kiss like would be bulb the has absolutely no movement whatsoever only that the body of the candle flicks I wanted to love them but sadly and inconveniently they are going back. Spare yourself if you’re looking for life like flames.',
    'Two of the candles do not work. I tried reaching out to the seller via their website listed with the product, but it does not exist.'
]

# Функция для предсказания сентимента с отслеживанием времени
def predict_sentiment(review):
    start_time = time.time()
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Review: {review}\nSentiment: {prediction[0]}\nExecution Time: {execution_time:.4f} seconds\n')

# Запуск программы и отслеживание времени выполнения
start_time = time.time()

# Вызов функции для каждого отзыва
for review in reviews_to_process:
    predict_sentiment(review)

end_time = time.time()
execution_time = end_time - start_time

print(f'Общее время выполнения программы: {execution_time:.4f} секунд')