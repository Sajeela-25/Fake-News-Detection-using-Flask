# Fake News Detection
This repository contains a *Fake News Detection* web application built using *Flask, **Python, and **Machine Learning. The project uses a trained model to classify news articles as either **Fake* or *Real*.

## Project Structure
fake-news-detection/
│── app.py                # Flask web application<br>
│── fakenews.py           # Jupyter Notebook (Dataset & Model Training)<br>
│── fake_news_model.pkl   # Trained machine learning model<br>
│── templates/            # HTML templates<br>
│   ├── fake.html         # Frontend UI for news input & prediction<br>
│── static/               # (Optional) CSS, JS, images<br>
│── README.md             # Project documentation<br>

### How to Run the Flask App
1. **Install Dependencies**
Ensure Python 3.x is installed, then install required packages:
```pip install flask scikit-learn```
2. **Run the Flask Server**
```python app.py```<br>
3. Open in Browser<br>
After running, the server will be available at:
```Running on http://127.0.0.1:5000/```

#### Project Files
• app.py → Flask app for loading the trained model & serving the UI.<br>
• fake.html → Web interface for news input and classification.<br>
• fakenews.py → Jupyter Notebook for data preprocessing, model training, and saving fake_news_model.pkl.<br>
• fake_news_model.pkl → Pre-trained model for fake news classification.

**How It Works**<br>
User enters a news article into the web app.<br>
The model processes the text and extracts features.<br>
**Prediction:** The model classifies the article as Fake or Real.<br>
Result is displayed on the web page.

**Future Improvements**<br>
Improve accuracy using deep learning models (LSTM, BERT).<br>
Enhance UI with Bootstrap or React.<br>
Deploy on Heroku or Render for public access.

**License**<br>
This project is licensed under the MIT License.
