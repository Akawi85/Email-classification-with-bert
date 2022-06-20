#!/usr/bin/env python3

import os, re
from subprocess import getstatusoutput

model_dir = './custom_model/'
config_json = "./custom_model/config.json"
pytorch_model = "./custom_model/pytorch_model.bin"
special_tokens = "./custom_model/special_tokens_map.json"
tokenizer_config = "./custom_model/tokenizer_config.json"
tokenizer = "./custom_model/tokenizer.json"
training_args = "./custom_model/training_args.bin"
vocab = "./custom_model/vocab.txt"
predict_script = "./predict.py"

def test_exists():
    """test if required files exists"""

    assert os.path.exists(model_dir)
    assert os.path.isfile(config_json)
    assert os.path.isfile(pytorch_model)
    assert os.path.isfile(special_tokens)
    assert os.path.isfile(tokenizer_config)
    assert os.path.isfile(tokenizer)
    assert os.path.isfile(training_args)
    assert os.path.isfile(vocab)
    assert os.path.isfile(predict_script)

def test_usage_help():
    """test that the help flag of the program works as expected"""

    for flag in ["-h", "--help"]:
        exit_code, output = getstatusoutput(f"{predict_script} {flag}")
        assert exit_code == 0
        assert re.match("usage", output, re.IGNORECASE)

def test_usage_email():
    """test that the email_text argument works as expected"""
    
    sample_mail = "I want to know if to share your email soon"
    exit_code, output = getstatusoutput(f"{predict_script} '{sample_mail}'")
    assert exit_code == 0
