# NA-404-App
Introduction:
We are living in a very politically charged environment these days and in this environment, people can easily express their thoughts and opinions on the internet freely without any fear. People say things on the internet which they don’t usually say in person and that can be because of multiple reasons.
The sentiments of the people should predict what should happen in the future because all politicians try to appease the people at least till before elections. 
Our program takes these opinions from Twitter, learns the general opinion and can then try to find out how many people support which political party. This can be very helpful to the media people of these respective parties when they want to find out about what people are saying about the party. 
Literature Review: 
Decision Trees:
We really didn’t look up for the research paper as we have studied this algorithm this semester but for the deep analysis and for the better understanding, we looked up some articles on the internet. This is one of the good one that we found. 
Random Forrest Classifiers:
Same was the case with Random forest Classifiers we looked at some articles that we found on the internet, we were more interested in this one because this is the optimization of the Decision tree as it removes the problem of overfitting and does pruning but we had to get a better idea how it should be done so, we found some helpful articles.  
Naïve Bayes:
We didn’t really at any research papers or work related to naïve bayes. We had done a lab related to it so we stuck with the program that we made in the lab.
Neural Network:
At first, we tried to employ a simple neural network for our project. That was giving us problems that were related to our dataset about how the model couldn’t process it. 
Then we tried looking for models that we specifically for sentiment analysis. This is where we found out about word embeddings and recurring neural networks. 
Most of these works had very complex models with multiple layers. This is because they had a dataset of almost 2 million reviews or text. We stuck to only three layers. One of the papers were using rmsprop as the optimizer but we stuck to adam. They kept the number of neurons for the input layer to a small amount but we decided to increase it because most of our tweets were long paragraphs. 
SVMs:
We’ve reviewed some web pages for implementing SVMs and also took help from lab of SVMs. What we reviewed from some web-based articles regarding our task of twitter-based sentiment analysis was that SVM is one of the widely used supervised machine learning techniques  for text classification. This systematic review helped us to analyze the latest work of sentiment analysis  with SVM using linear and poly kernels as well as provide them a baseline for future trends and comparisons.

Dataset:
The Data we are using is non-numeric. We initially wanted a dataset only related to specific Pakistani Politics. So, we had to create it on our own. We made our dataset from the scratch.

•	Initially total tweets extracted were 5000.
•	Manually removing the duplicates and empty ones we cut it down to approximately 4000 tweets.
•	After applying pre-processing techniques, we now have approximately 3000 tweets.
•	Total Keywords = 40
•	Total features = 8
•	Useful Features = 3
•	We had 2 output columns (Response & Party)
•	Response had three classes: Positive, Negative & Neutral
•	Party had 4 Classes that are: PTI, PMLN, PPP, Neutral

Step 1: We made a twitter developer account and waited for the access permission.
Step 2: We used the Tweety API that helps you in extracting tweets for specific keywords, hash tags or for specific dates.
Step 3: We decided approximately 40 keywords against which we extracted the tweets.
Step 4: Writing those tweets into CSV files.
Step 5: We manually read each tweet and labeled the output columns of which party it belongs to or what type of sentiment it shows
Step 6: We manually cleaned the data, removed the extra words and quotation marks and images that were not useful.
Step 7: We applied built-in functions for cleaning that data. 

PRE- PROCESSING STEPS:
1.	Remove Punctuations, Urls, stop words, converting it to lower case etc.
2.	Mapping Tweets to a Vector Space. 
3.	Converted the Non-Numeric data into Numeric so we can process it and run it on our models.

Baseline:
The Baseline of our project NA-404 was to do sentimental analysis.  It can provide you the bias free result that you want to see. Politics is always a hot topic to discuss so that is why we solely focused on it. We made two output columns (Response and Party). The User can just input the tweets he wants to look up for and can get the result that whether this tweet supports PTI, PMLN, PPP or is it just a normal or Neutral tweet. He/she can also check what emotion this tweet reflects.
Base line of NA-404 was using ML classification models to classify the result correctly. We used six ML Models. They are “Linear SVM, Poly SVM, Decision tree, random forest classifiers, Naïve Bayes and Neural Networks “
Main Approach: 
We have used 6 models in total. The first few steps are the same for all of them. Our dataset is in string form with many unnecessary values in it. So, we have a few general cleaning functions (for removal of punctuation, URLs, stop words etc). The output after this is still in string form so we vectorize it. We use CountVectorizer for this. This basically converts string into numeric data. The numeric data is a sparce matrix. This matrix shows the number of times a word comes in a sentence. The rows of the sparce matrix will correspond to the number of tweets in our dataset and the columns will correspond to the number of unique words in a tweet.  
1.	SVMs (Support Vector Machines): SVMs have been used to do text-classification analysis, and since our task is also text classification so we used SVMs. One important part of SVMs are kernels and gamma value which is passed as a parameter to SVC Fit model function. This gamma value is very important and if we choose this large then it’ll overfit our model so we choose 0.2 after some hit and trials. And for kernels, although there are several widely used kernel functions, a carefully designed kernel will help to improve accuracy of SVMs. Therefore, we used two different kernel designs one is linear kernel and other is poly-kernel 
SVM (linear kernel): It is used to classify linearly separatable data and it gives 71% accuracy with our dataset that means that this kernel performs well for our dataset.
SVM (poly kernel): we can think of the polynomial kernel as a processor to generate new features by applying polynomial combinations and it gives 69% accuracy with our dataset which is not good as lin-kernel but still better try to classify.

2.	Naïve Bayes:
The naïve bayes model is based on the bayes rule which uses the probability of occurrence of an event to judge the occurrence of another event related to it.
 
The Multinomial Naïve Bayes Classifier is appropriate for features that represent counts or count rates. Since our dataset also revolves around the same principle as we are distinguishing tweets based on the type of words that our appearing, the associated response, how many times these words appear etc, we have used this classifier. 
We split our vectorized data into training and testing sets (70-30 ratio) and train the model using our training data. 
3.	Neural Network:
The one preprocessing step different here than the remaining models are that instead of vectorizing the string data and converting it into a sparse matrix, we used word embedding. 
It basically maps the data onto a vector space and allows similar meaning data to have a similar representation. 
The word embedding in our case is done from scratch instead of using a pretrained model. This is because our dataset has many roman urdu tweets as well and we thought that the accuracy of our model will be compromised if we use a pretrained model.
After trying different types of neural networks, we eventually settled with a recurring neural network. This is because our data was text based. When we read anything, we remember what we are reading and thus are able to get a cohesive picture. That is what RNNs do as well. They try to make connections between the data that they have learnt. 
Layers:
1.	The embedded layer as the input layer.
2.	An LSTM layer as the hidden layer
3.	An output layer with softmax activation.
Response:
Training Accuracy: 97%
Validation Accuracy: 75%
Party:
Training Accuracy: 98%
Validation Accuracy: 76%

4.	Decision Tree:
The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision. The main part to is to find the root node that is the most suitable one and then start splitting and the decision is made at each node on the basis of the decision made at last step and at the leaf node we get the final decision. This works good in case of small dataset and that is why we used this approach. 
•	Accuracy for Response: 76%
•	Accuracy for Party: 78%

5.	Random Forrest Classifiers:
It can perform both regression and classification tasks. A random forest produces good predictions that can be understood easily. It can handle large datasets efficiently. It basically consists of many decision trees and it does bagging and chooses the most suitable results. Also, one of the reasons we used this classifier was because it ignores the clash of over fitting and does pruning
•	Accuracy for Response:  80.6%
•	Accuracy for Party: 81.4%
Evaluation Metrics:
As a classification problem, our application NA-404 uses evaluation metrics called “classification report” . Precision, recall, and F1 score are the metrics associated with the classification report.
	Precision: is the ability of a classifier not to label an instance positive that is actually negative.
Precision – accuracy of positive predictions.
Precision = TP/ (TP + FP)
	Recall: is the ability of a classifier to find all positive instances.
Recall – fraction of positives that were correctly identified.
Recall = TP/ (TP + FN)
	F1 Score: what percent of positives that were correct.
F1 Score = 2*(Recall * Precision) / (Recall + Precision)
	Accuracy: Accuracy simply measures how often the classifier correctly predicts. We can define accuracy as the ratio of the number of correct predictions and the total number of predictions.
 

One sample classification report for the SVM classifier is shown below:
 
Result and Analysis:
Our Results were surprisingly very accurate. It not only gave correct results for English tweets but roman Urdu as well. 
Our main approach was to get the tweet classified in two features i.e. Party (having 4 classes) and Responses (having 3 classes). It not only classified testing data on basis of training data, but also figured out the overall result. 
We were first implementing SVM model on our raw dataset which gave us very low results. Then after implementing preprocessing steps, our accuracy increased from 59 to 70 percent which is a good ratio.
According to accuracy, Neural Network gives the best result up to 97.7% accuracy, which is due to our small dataset. By right accuracy, we mean that our input is rightly classified in terms of both response and party.
Error Analysis:
 
All models we experimented to do our twitter classification are shown with respective accuracy.
According to our results, Naïve Based had lesser accuracy than Neural Networks. Although Naïve Based model should have performed best, this may be due to our smaller dataset.
Additionally, Neural Network Model had over fitted our dataset which overshadowed our Naive Based model’s results.
Future Work: 
This section can be short, but please include some ideas about how you could improve your model if you had more time. This can also include any challenges you're running into and how you might fix it.
1.	We could have expanded our dataset; taken data every day for a week to get a more thorough dataset which will include new relevant news.
2.	Instead of just using the tweet as an input feature, we could also have taken the place where the person had tweeted. This would have given an extra feature of what sentiments do people have from different areas of the world.
3.	We could have added new output features as well such as the specific topic the tweet was addressing. In the party section, we only added the three major political parties. We could have added the other ones as well.
4.	We could have made an interface where people could easily give the data (in bulk or a single tweet) to the program and the program would have given the stats about the data.

References/Literature:
  https://thesai.org/Publications/ViewPaper?Volume=9&Issue=2&Code=IJACSA&SerialNo=26
 https://www.researchgate.net/publication/323536661_Sentiment_Analysis_using_SVM_A_Systematic_Literature_Review

  https://www.sciencedirect.com/topics/computer-science/evaluation-metric 

https://towardsdatascience.com/decision-tree-in-machine-learning-e380942a4c96#:~:text=Decision%20Trees%20are%20a%20non,values%20are%20called%20classification%20trees.

https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/

https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91



