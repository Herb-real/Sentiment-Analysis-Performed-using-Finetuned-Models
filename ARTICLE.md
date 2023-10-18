**Sentiment Analysis Explained?**

Sentiment analysis, a natural language processing approach, is employed to ascertain the emotional underpinning of a collection of words, phrases, or sentences.

In recent years, sentiment analysis has surged in prominence across diverse domains, including customer service, brand surveillance, and the scrutiny of social media content.

The utilization of pre-trained language models like BERT (Bidirectional Encoder Representations from Transformers) has streamlined the execution of sentiment analysis tasks. This article delves into the process of fine-tuning a pre-trained BERT model for sentiment analysis using Hugging Face and subsequently uploading it to the Hugging Face model repository.

**What is the rationale behind choosing Hugging Face?**

Hugging Face serves as a comprehensive platform equipped with an array of tools and resources tailored for natural language processing (NLP) and machine learning tasks.

Its user-friendly interface and an extensive array of pre-trained models, datasets, and libraries make it a valuable resource for data analysts, developers, and researchers.

Hugging Face boasts an extensive collection of pre-trained models specifically fine-tuned for various NLP tasks like text classification, sentiment analysis, named entity recognition, and machine translation.

These models not only offer a convenient starting point for your projects but also significantly reduce the time and effort needed for training models from the ground up

For this particular project, I recommend enrolling in this course to explores the ins and outs of natural language processing (NLP) while leveraging the Hugging Face ecosystem. To get started, please visit the website and register to unlock access to all the platform’s features.

Please, go to the website and sign in to access all the features of the platform.

Read more about Text classification with Hugging Face

**Leveraging GPU Runtime within Google Colab**

Prior to delving into the code, it is crucial to comprehend the reasons behind employing GPU runtime on Google Colab is beneficial.

GPU, which is an abbreviation for Graphics Processing Unit, is robust hardware developed for managing intricate graphics and computations.

Hugging Face models rely on deep learning, which necessitates substantial GPU computational power for training.

Feel free to utilize Colab to accomplish this, or alternatively, consider a different GPU cloud provider or a local machine equipped with an NVIDIA GPU.

In our project, we harnessed the GPU runtime on Google Colab to expedite the training process.

To gain access to a GPU within Google Colab, you simply need to opt for the GPU runtime environment when initiating a new notebook.

This choice enables us to make the most of the GPU’s capabilities and significantly accelerate our training tasks.


changing runtime to GPU

**Setup**
With a grasp of the GPU’s significance, let’s move on to the code. Our first step involves the installation of the transformers library, a Python library created by Hugging Face.

This library offers a range of pre-trained models and utilities for fine-tuning them. Additionally, we will install any other necessary dependencies.

!pip install transformers
!pip install datasets
!pip install --upgrade accelerate
!pip install sentencepiece
Next, we import the necessary libraries and load the dataset. In this project, we will use the dataset from this Zindi Challenge , which can be downloaded here.

import huggingface_hub # Importing the huggingface_hub library for model sharing and versioning
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
Furthermore, we generated a PyTorch dataset. These PyTorch datasets adhere to a standardized format, offering enhanced efficiency and convenience for our machine learning procedures.

Adhering to this dataset structure guarantees uniform data handling and effortless integration with various other PyTorch functionalities.

# Split the train data => {train, eval}
train, eval = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
Preprocessing
Moving forward, we undertake the text data cleansing and tokenization process.

Machine learning models exclusively operate with numerical data. Tokenization is a vital step in translating text into numeric representations, often referred to as word embeddings.

These embeddings consist of compact vector representations that encapsulate the semantic significance and associations among words.

These representations empower machines to grasp contextual information and analogies between words, thereby simplifying more advanced natural language processing tasks.

To streamline our preprocessing efforts, we will employ two widely used functions for text preparation and label conversion.


The preprocess function modifies text by replacing usernames and links with placeholders.The transform_labels function converts labels from a dictionary format to a numerical representation.

Tokenization
We define the checkpoint variable, which holds the name or identifier of the pre-trained model we want to use. In this case, it’s the “cardiffnlp/twitter-xlm-roberta-base-sentiment” model.
tokenizer = AutoTokenizer.from_pretrained(checkpoint): We create a tokenizer object using the AutoTokenizer class from the transformers library. The tokenizer is responsible for converting text data into numerical tokens that the model can understand.
def tokenize_data(example): We define a function called tokenize_data that takes an example from the dataset as input. This function uses the tokenizer to tokenize the text in the example, applying padding to ensure all inputs have the same length.
dataset = dataset.map(tokenize_data, batched=True): We apply the tokenize_data function to the entire dataset using the map method. This transforms the text data in the ‘safe_text’ column into tokenized representations, effectively preparing the data for model consumption. The batched=True parameter indicates that the mapping operation should be applied in batches for efficiency.
remove_columns = [‘tweet_id’, ‘label’, ‘safe_text’, ‘agreement’]: We create a list called remove_columns that contains the names of the columns we want to remove from the dataset.
dataset = dataset.map(transform_labels, remove_columns=remove_columns): We apply another transformation to the dataset using the map method. This time, we use the transform_labels function to transform the labels in the dataset, mapping them to numerical values. Additionally, we remove the columns specified in the remove_columns list, effectively discarding them from the dataset.
By tokenizing the text data and transforming the labels while removing unnecessary columns, we preprocess the dataset to prepare it for training or evaluation with the sentiment analysis model.

Training
Now that we have our preprocessed data, we can fine-tune the pre-trained model for sentiment analysis. We will first specify our training parameters.

# Configure the trianing parameters like `num_train_epochs`: 
# the number of time the model will repeat the training loop over the dataset
training_args = TrainingArguments("test_trainer", 
                                  num_train_epochs=10, 
                                  load_best_model_at_end=True, 
                                  save_strategy='epoch',
                                  evaluation_strategy='epoch',
                                  logging_strategy='epoch',
                                  logging_steps=100,
                                  per_device_train_batch_size=16,
                                  )
We set the hyperparameters for training the model, such as the number of epochs, batch size, and learning rate.

We’ll load a pre-trained model, shuffle the data, and then define the evaluation metric. In this case, we are using rmse.

# Loading a pretrain model while specifying the number of labels in our dataset for fine-tuning
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
train_dataset = dataset['train'].shuffle(seed=24) 
eval_dataset = dataset['eval'].shuffle(seed=24) def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"rmse": mean_squared_error(labels, predictions, squared=False)}
We can easily train and evaluate our model using the provided training and evaluation datasets by initialising the Trainer object with the parameters below.

The Trainer class handles the training loop, optimization, logging, and evaluation, making it easier for us to focus on model development and analysis.

trainer = Trainer(
    model,
    training_args, 
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
Finally, we can train our model using.

trainer.train()

And launch the final evaluation using.

trainer.evaluate()

Next Steps
Wondering where to go from here? The next step would be to deploy your model using Streamlit or Gradio, for example.

This would be a web application that your users can interact with in order to make predictions. Here are screenshots of two web apps built with the model we just finetuned.



**Conclusion**
In summary, we have refined a pre-existing model for sentiment analysis using the Hugging Face library on a dataset. Following ten training epochs, the model attained a validation set root mean square error (RMSE) score of 0.659.

Find all the code for this project below.

GitHub - Herb-real/Sentiment-Analysis-Performed-using-Finetuned-Models: Finetuned Models for…
Finetuned Models for Natural Language Processing. Contribute to…
github.com