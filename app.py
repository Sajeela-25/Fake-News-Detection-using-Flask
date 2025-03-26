from flask import Flask, request, render_template
import pickle
import os

# Debugging: Print working directory
print("Current working directory:", os.getcwd())

# Load trained model and vectorizer
with open("fake_news_model.pkl", 'rb') as file:
    vectorizer, model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('fake.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_article']  # Match the form name in HTML
    news_tfidf = vectorizer.transform([news_text])
    prediction = model.predict(news_tfidf)[0]
    result = "Fake News" if prediction == 1 else "Real News"
    return render_template('fake.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
