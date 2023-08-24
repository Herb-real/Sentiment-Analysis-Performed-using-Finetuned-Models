# Sentiment Analysis Performed using Finetuned Models
______________________________________________________________________________________

# Introduction
______________________________________________________________________________________
Works are underway to create a COVID-19 vaccine, building on the success of vaccines in mitigating the impact of diseases like measles and the flu. These medical interventions have significantly reduced the occurrence of illnesses and fatalities, playing a pivotal role in preserving innumerable lives worldwide. Regrettably, the 'anti-vaxxer' movement has caused vaccination rates to drop in certain nations, causing the resurgence of erstwhile diseases.

Although it might take several months before COVID-19 vaccines are universally accessible, it's crucial to presently and particularly in the future, when COVID-19 vaccines become accessible, keep a close watch on public attitudes toward vaccinations. The presence of anti-vaccination sentiments could critically jeopardize the sustained global endeavor to effectively manage COVID-19 in the long run.

**The objective of this challenge is to develop a machine learning model to assess if a Twitter post related to vaccinations is positive, neutral, or negative. This model will be deployed using streamlit on a Docker Container.**

# Dataset
_______________________________________________________________________________________
Tweets have been classified as pro-vaccine (1), neutral (0) or anti-vaccine (-1). The tweets have had usernames and web addresses removed.

Variable definition:

tweet_id: Unique identifier of the tweet

safe_tweet: Text contained in the tweet. Some sensitive information has been removed like usernames and urls

label: Sentiment of the tweet (-1 for negative, 0 for neutral, 1 for positive)

agreement: The tweets were labeled by three people. Agreement indicates the percentage of the three reviewers that agreed on the given label. You may use this column in your training, but agreement data will not be shared for the test set.

Files available for download are:

Train.csv - Labelled tweets on which to train your model

Test.csv - Tweets that you must classify using your trained model

SampleSubmission.csv - is an example of what your submission file should look like. The order of the rows does not matter, but the names of the ID must be correct. Values in the 'label' column should range between -1 and 1.

NLP_Primer_twitter_challenge.ipynb - is a starter notebook to help you make your first submission on this challenge.

# Setup
______________________________________________________________________________________________
Fork this repo and run the notebook on Google Colab. The Hugging face models are Deep Learning based, so will need a lot of computational GPU power to train them. Please use Colab to do it, or your other GPU cloud provider, or a local machine having NVIDIA GPU.

Note that Google Colab sessions have time limits and may disconnect after a period of inactivity. However, you can save your progress and re-establish the connection to the GPU when needed.

Hugging Face is an open-source and platform provider of machine learning technologies. You can use install their package to access some interesting pre-built models to use them directly or to fine-tune (retrain it on your dataset leveraging the prior knowledge coming with the first training), then host your trained models on the platform, so that you may use them later on other devices and apps.

Please, go to the website and sign-in to access all the features of the platform.

Read more about Text classification with Hugging Face

# Evaluation
________________________________________________________________________________________________
The evaluation metric for this challenge is the Root Mean Squared Error.

# Screenshot
_________________________________________________________________________________________________



# Resources
_________________________________________________________________________________________________
1. Quick intro to NLP
2. Getting Started With Hugging Face in 15 Minutes
3. Fine-tuning a Neural Network explained
4. Fine-Tuning-DistilBert - Hugging Face Transformer for Poem Sentiment  
5. Prediction | NLP
6. Introduction to NLP: Playlist