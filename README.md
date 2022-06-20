# Email Classification with BERT
This project finetunes a pretrained text classification `bert-base-uncased` model on an weirdly small dataset in order to implement email classification on the provided dataset.
### Background to the problem 
The problem assumes that Mary, a college student adviser helps students solve timetable clashes over email.
Suppose Mary is so helpful that students want to share her email address with their friends.  
To ensure students know they can share her email address, Mary has updated her signature to say, If you found this conversation helpful, feel free to share my email address with your friends.  
With the updated signature, Mary now gets about 250 emails from students a week either asking if they can share her email address with other students, or saying they’ve shared her email address with other students.

Most of the emails Mary gets are unpunctuated one liners, like the below:  
- "Can I share your email"  
- "I will share your email"
- "I shall share your email"
- "I've shared your email"
- "May I share your email"
- "Should I share your email"
- "I already shared the email"
- "I've just shared your email"
- "Am I allowed to share your email"
- "Am I able to share your email"
- "I am able to share your email"
- "Will you help my friends if I share your email with them?"

Mary has asked the college’s IT department to help her find or build some kind of filter that labels emails of this kind as either:  
**Student has shared**  or
**Student wants to know if can share**

### [Data Preparation, Tokenization and Modelling](https://github.com/Akawi85/Email-classification-with-bert/blob/main/train_model.ipynb)
##### Preparing the Dataset
From the problem statement, the sample emails provided by Mary constituted solely the dataset used for training this model. The sample email are just eleven (11) and since Mary is classifying the email into just two exclusive classes, a binary classification approach was adopted in executing the task.  
Since the dataset is so small, the correct class labels were manually allocated to each example email and loaded as a pandas dataframe.
See [the notebook](https://github.com/Akawi85/Email-classification-with-bert/blob/main/train_model.ipynb) for details on approaches taken in preparing the dataset.

##### Tokenization and Modelling
The curated/preprocessed dataset was tokenized using the pretrained `bert-base-uncased` checkpoint.  
The same pretrained checkpoint was used for training the model using the `AutoModelForSequenceClassification` API.  
The model trained on the sample emails dataset for 20 epochs and achieved an `accuracy` and `f1` scores of `1.0` and `1.0` on the training dataset respectively.  
This metrics are highly flawed because we are evaluating on the same training set and can be ascribed to overfitting, but since we have no test set to evaluate on given the size of the sample data, we can assume that for the model to overfit at 20 epochs, it actually did well in learning to classify the training dataset to the extent of overfitting.  
All the model training checkpoints and weights were downloaded and saved in the [custom_model](https://github.com/Akawi85/Email-classification-with-bert/tree/main/custom_model) directory.

The [train_model.ipynb](https://github.com/Akawi85/Email-classification-with-bert/blob/main/train_model.ipynb) script shows how all of the data preparation, tokenization and modelling steps were implemented.

### Creating a Command Line Program that Predicts Email Class
The ultimate goal of the project is to create an easy to use program that will help classify Mary's emails into the appropriate classes.  
To do this, a program was created using the trained model, that accepts user inputs from the command line (email body) and outputs the class of the input as shown below:

![program_usage](./img/program_test_1.png)

The [predict.py](https://github.com/Akawi85/Email-classification-with-bert/blob/main/predict.py) script shows how the program was implemented in the command line as well as how errors and other input format specifications were handled when passing arguments to the program.  

### [Testing](https://github.com/Akawi85/Email-classification-with-bert/blob/main/test.py)
The [test.py](https://github.com/Akawi85/Email-classification-with-bert/blob/main/test.py) script implements unit testing of the various functionalities of the program.

### Running the program on your machine
- Clone the repo into a folder on your local machine. You can do that by running this command  
`https://github.com/Akawi85/Email-classification-with-bert.git` in your terminal.
  - *Note that the model used for prediction is about `418` megabytes, therefore you may need to install the git extension **git lfs** in order to get the actual model file. Follow [this guide](https://www.atlassian.com/git/tutorials/git-lfs) for a thorough walkthrough on how to implement this*
- Move into the repo directory and create and activate a python virtual environment in order to install all the program requirements in an isolated environment. If you're using Windows Subsystem for Linux (WSL), I recommend [this document](https://docs.google.com/document/d/19IpozHrM38HzVSI4PjwRFJSNeLdcceUKg98fr2Db-DQ/edit?usp=sharing), where I outlined detailed steps for creating python virtual environments in WSL.
- Install the program requirement files in the newly created virtual environment by running the command:  
`pip install -r requirements.txt`
- Run the program by calling the `predict.py` script and providing the required argument `email_text`. An example of expected input is:  
`python3 predict.py "I would love to share your email with my friends, can I go ahead"`  

  - ***Notes***  
    - *The that the program expects you to pass the content of the email you wish to classify as required argument enclosed in single or double quotes as shown above*
    - *You can seek help for running the program by calling the help flags `-h` or `--help` as follows:*
     `python3 predict.py -h`
- Run program tests as follows:  
`pytest test.py`


***PS: I created the command line program in a WSL environment, therefore you may experience some slight fails when implementing the program following the above guidelines. But a simple google search should suffice in those cases***

### Areas of further improvements
1. Collect more data for model training, testing and validation
2. Perform data augmentation for generating synthetic datasets and increase samples
3. Write more advanced unit tests for the program
4. Create a web interface as a replacement for command line implementation
5. Dockerize the program to eliminate operating system dependencies.
