#!/usr/bin/env python3

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import argparse
import os
import sys

# disable parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# get the directory where the model was saved to
model = AutoModelForSequenceClassification.from_pretrained('./custom_model/')

# load the tokenizer by pointing to the same directory as the pretrained model
tokenizer = AutoTokenizer.from_pretrained('./custom_model/')

#---------------------------------------------------------------------------------------------------
def get_args():
    """
    Get command line arguments
    """

    parser = argparse.ArgumentParser(description="Classify students emails based on mail content",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('email_text',
                        type=str,
                        help='Enter the content of the email you want to classify in enclosed quotes')

    args = parser.parse_args()

    return args

#-------------------------------------------------------------------------------------------------------
def main():

    args = get_args()
    text = args.email_text

    # check that the required argument contains relevant words for better prediction
    if ('email' in text) | ('e-mail' in text) | ("mail" in text) | ('mailed' in text) \
        | ('share' in text) | ('shared' in text) | ("sharing" in text):

        classifier = pipeline(task='text-classification', model=model, tokenizer=tokenizer)
        predicted = classifier(text)
        predicted_label = predicted[0]['label']

        print(' The mail to classify is: \n', ''.join(text), '\n --------------------------------------\
------------------------------------------------')

        show_pred_class = np.where(predicted_label == 'LABEL_1',
                                'Student wants to know if can share',
                                    'Student has shared')
            
        print('\n The mail is classified as: \n', show_pred_class)

    else:
        print("please ensure your email body contains at least one of the following keywords: \
             \n email \n e-mail \n share \n shared \n sharing")

# ----------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()