from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import argparse
import os

# disable parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# get the directory where the model was saved to
model = AutoModelForSequenceClassification.from_pretrained('./custom_model/')

# load the tokenizer by pointing to the same directory as the pretrained model
tokenizer = AutoTokenizer.from_pretrained('./custom_model/')

#------------------------------------------------------------------
def get_args():
    """
    Get command line arguments
    """

    parser = argparse.ArgumentParser(description="Classify students emails based on mail content",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('email_text',
                        nargs='*',
                        type=str,
                        metavar='email',
                        help='Enter the email body you want to classify')

    args = parser.parse_args()

    return args


def main():

    args = get_args()
    text = args.email_text

    print(' Mail to classify is: \n', ''.join(text), '\n -----------------------------------------------------------------')
    classifier = pipeline(task='text-classification', model=model, tokenizer=tokenizer)
    predicted = classifier(text)
    predicted_label = predicted[0]['label']
    show_pred_class = np.where(predicted_label == 'LABEL_1',
                               'Student wants to know if can share',
                               'Student has shared')
    
    print('\n The mail is classified as: \n', show_pred_class)
    return predicted

# --------------------------------------------------
if __name__ == '__main__':
    main()