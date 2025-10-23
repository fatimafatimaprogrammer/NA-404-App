# 🗳️ NA-404 App — Twitter-Based Political Sentiment Analysis

## 📘 Introduction  
In today’s politically charged environment, people freely express their thoughts and opinions on the internet — especially on platforms like **Twitter**. These opinions reflect public sentiment and can serve as an indicator of future political trends.  

**NA-404 App** leverages **Machine Learning (ML)** models to analyze Twitter data and determine the **sentiment** (Positive, Negative, Neutral) and **political affiliation** (PTI, PMLN, PPP, or Neutral) expressed in tweets.  

This tool can help political analysts, journalists, and researchers understand public perception and sentiment toward different political parties.

---

## 🧠 Project Overview
- **Domain:** Political Sentiment Analysis  
- **Objective:** Predict which political party a tweet supports and the sentiment it conveys.  
- **Input:** Tweets fetched from Twitter using Tweepy API  
- **Output:** Predicted political affiliation + Sentiment type  
- **Language:** Python  
- **Techniques Used:** NLP, Text Classification, Machine Learning  

---

## 📚 Literature Review
We explored multiple ML techniques and algorithms to find the most suitable ones for our dataset and classification goal.

| Algorithm | Description | Accuracy | Reference |
|------------|-------------|-----------|------------|
| **Decision Trees** | Simple rule-based model for classification tasks. Works best for smaller datasets. | Response: 76% <br> Party: 78% | [Decision Trees - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/) |
| **Random Forest Classifier** | Ensemble of Decision Trees; reduces overfitting and improves accuracy. | Response: 80.6% <br> Party: 81.4% | [Understanding Random Forest](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/) |
| **Naïve Bayes** | Probabilistic model using Bayes’ theorem; suitable for text classification. | Moderate | Used from lab implementation |
| **Support Vector Machines (SVM)** | Used for text classification; tested with linear and polynomial kernels. | Linear: 71% <br> Poly: 69% | [SVM Systematic Review](https://thesai.org/Publications/ViewPaper?Volume=9&Issue=2&Code=IJACSA&SerialNo=26) |
| **Neural Networks (RNN + LSTM)** | Deep learning model trained using word embeddings; performed best overall. | Response: 97% <br> Party: 98% | [Sentiment Analysis with Keras](https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91) |

---

## 🗂️ Dataset Details
Since no suitable dataset existed for **Pakistani political tweets**, we created one manually.

| Step | Description |
|------|--------------|
| 1 | Created a Twitter Developer account and fetched tweets using **Tweepy API**. |
| 2 | Extracted **~5000 tweets** based on ~40 political keywords. |
| 3 | After cleaning and removing duplicates → **~3000 tweets** remained. |
| 4 | Manually labeled each tweet with **Sentiment (Positive/Negative/Neutral)** and **Party (PTI/PMLN/PPP/Neutral)**. |
| 5 | Cleaned data using preprocessing functions (punctuation removal, lowercasing, stopword removal, etc.). |

**Dataset Summary:**
- Total Tweets: 3000  
- Keywords: 40  
- Features: 8 (3 useful)  
- Output Columns: 2 → `Response`, `Party`  
- Languages: English + Roman Urdu  

---

## 🧹 Preprocessing Steps
To prepare the text data for analysis:
1. Removed punctuations, URLs, mentions, and stopwords.  
2. Converted all text to lowercase.  
3. Tokenized and vectorized tweets using **CountVectorizer**.  
4. Converted textual data into a **sparse numeric matrix** for ML model input.  
5. For Neural Networks — applied **Word Embedding** for semantic representation.

---

## 🧮 Machine Learning Models Used

### 🧩 SVM (Support Vector Machines)
- Used **linear** and **polynomial kernels**.  
- Tuned **gamma = 0.2** after several trials.  
- Linear kernel performed best with **71% accuracy**.  

### 🧩 Naïve Bayes
- Implemented **Multinomial Naïve Bayes Classifier**.  
- Trained with 70-30 train-test split.  
- Moderate performance, limited by dataset size.

### 🧩 Decision Trees
- Built hierarchical decision models based on entropy.  
- **Accuracy:** Response = 76%, Party = 78%.  
- Effective for small datasets.

### 🧩 Random Forest Classifier
- Ensemble of multiple Decision Trees (bagging + pruning).  
- **Accuracy:** Response = 80.6%, Party = 81.4%.  
- Reduced overfitting and improved stability.

### 🧩 Neural Network (RNN + LSTM)
- Used **Word Embeddings** and **LSTM layers** for sequence learning.  
- Architecture:
  - Embedding layer (Input)
  - LSTM hidden layer
  - Softmax output layer  
- **Accuracy:**  
  - Response: 97% (train) / 75% (validation)  
  - Party: 98% (train) / 76% (validation)

---

## 📊 Evaluation Metrics
Used **Classification Report** with:
- **Precision:** TP / (TP + FP)  
- **Recall:** TP / (TP + FN)  
- **F1 Score:** 2 × (Precision × Recall) / (Precision + Recall)  
- **Accuracy:** Correct predictions / Total predictions  

---

## 📈 Results & Analysis
- The model correctly classified both **English** and **Roman Urdu** tweets.  
- Best performance achieved with **Neural Network (RNN)** model.  
- Preprocessing improved accuracy significantly (from 59% → 70%).  
- Random Forest and Decision Tree models gave consistent results with smaller datasets.  

---

## 🚀 Baseline & Approach
**Baseline:**  
To perform unbiased sentiment classification for political tweets in Pakistan.  

**Approach:**
- Use ML classification algorithms (SVM, Naïve Bayes, Decision Tree, Random Forest, Neural Network).  
- Classify tweets into:
  - **Response:** Positive / Negative / Neutral  
  - **Party:** PTI / PMLN / PPP / Neutral  

---

## 🔮 Future Work
- Expand dataset — collect tweets daily for more diversity.  
- Include **geo-location** of tweets to analyze regional sentiments.  
- Add additional political parties and topics.  
- Build a **user interface (UI)** for easier input and real-time predictions.  
- Integrate **live sentiment dashboards** for media and research use.  

---

## 📚 References  
- [Sentiment Analysis using SVM – IJACSA](https://thesai.org/Publications/ViewPaper?Volume=9&Issue=2&Code=IJACSA&SerialNo=26)  
- [Decision Tree in Machine Learning – Towards Data Science](https://towardsdatascience.com/decision-tree-in-machine-learning-e380942a4c96#:~:text=Decision%20Trees%20are%20a%20non,values%20are%20called%20classification%20trees.)  
- [Understanding Random Forest – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)  
- [Evaluation Metrics – ScienceDirect](https://www.sciencedirect.com/topics/computer-science/evaluation-metric)  
- [Sentiment Analysis with Deep Learning and Keras – Towards Data Science](https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91)

---

## 🏷️ Suggested Repository Title
**NA-404: Twitter-Based Political Sentiment Analysis using Machine Learning**

---

## 📅 Academic Context
This project was developed as part of a **Data Science / Machine Learning semester project** to analyze political sentiment from real-world Twitter data.
