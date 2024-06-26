# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/197ugalmlFnBH-JZ1B8DFFZA0FN6uZjQc
"""

import numpy as np
import pandas as pd
import os
import nltk
from nltk import word_tokenize
import re
import string
import unicodedata
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from camel_tools.ner import NERecognizer
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from nltk.stem import PorterStemmer

words = ['أَنْ', 'حاسَب','عائله', 'اب', 'ام', 'انا', 'طالب', 'كلية', 'جامعه', 'المنصورة', 'حاسبات',
         'معلومات', 'اهلا', 'يحب', 'في', 'عمل', 'شكرا', 'و']
character = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '100', '1000',
             'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س',
             'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ة']

# Initialize the Camel Tools components
db = MorphologyDB.builtin_db()
analyzer = Analyzer(db)
stemmer = PorterStemmer()


def clean_text(text):
    punctuations = string.punctuation + "،"
    punct_free = "".join([char for char in text if char not in punctuations])
    return punct_free

def remove_plus(word):
    word = word.replace("+", "")
    return word

def remove_ta_marbuta(word):
    if word.endswith('ة'):
        word = word[:-1]
    return word

def remove_al(word):
    analyses = analyzer.analyze(word)
    if analyses:
        for analysis in analyses:
            if 'lex' in analysis:
                return analysis['lex']
    return word

def tokenize_arabic_text(text):
    cleaned = clean_text(text)
    tokens = simple_word_tokenize(cleaned)
    return tokens

def get_lexical_forms(tokens):
    lexical_forms = []
    for token in tokens:
        analyses = analyzer.analyze(token)
        if analyses:
            lex = analyses[0]['lex']
            lexical_forms.append(lex)
    return lexical_forms

def search_lexical_forms(lexical_forms, predefined_list):
    found_words = []
    not_found_lexical_forms = []
    for lex in lexical_forms:
        lex = dediac_ar(lex)
        lex = remove_al(lex)
        lex = remove_ta_marbuta(lex)
        lex = remove_plus(lex)
        found_match = False
        for word in predefined_list:
            #word = dediac_ar(word)
            cleaned_word = clean_text(word)
            cleaned_word = remove_al(cleaned_word)
            cleaned_word = remove_ta_marbuta(cleaned_word)
            cleaned_word = remove_plus(cleaned_word)
            if lex == stemmer.stem(cleaned_word):
                if cleaned_word == 'حاسَب':
                    found_words.append("حاسبات")
                elif cleaned_word == 'أَنْ':
                    found_words.append("انا")
                else:
                    found_words.append(word)
                    found_match = True
                    break
        if not found_match:
            not_found_lexical_forms.append(lex)

    return found_words, not_found_lexical_forms

def process_arabic_text(text, predefined_list):
    tokens = tokenize_arabic_text(text)
    lexical_forms = get_lexical_forms(tokens)
    found_lexical_forms, not_found_lexical_forms = search_lexical_forms(lexical_forms, predefined_list)
    return found_lexical_forms


text = "أنا"
result = process_arabic_text(text, words)
print(result)