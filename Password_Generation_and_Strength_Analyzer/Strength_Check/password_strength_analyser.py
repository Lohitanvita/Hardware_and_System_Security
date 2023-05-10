import string
import pandas as pd
import numpy as np
import random
import datetime as dt
import re
from colored import fg


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


# Tokenization is the process of breaking text data into simpler characters called tokens.
# We will break our password text into word tokens which we will use as input for our model.


def tokenizing(X_predict, all_passwords, y_labels):
    # Machine learning models do not comprehend text.
    # We therefore need to further convert the word tokens to numeric data.
    def create_tokens(f):
        tokens = []
        for i in f:
            tokens.append(i)
        return tokens

    vectorizer = TfidfVectorizer(tokenizer=create_tokens)
    X = vectorizer.fit_transform(all_passwords)
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.27, random_state=42)
    logit = LogisticRegression(penalty='l2', multi_class='ovr')
    logit.fit(X_train, y_train)
    X_predict = vectorizer.transform(X_predict)
    y_Predict = logit.predict(X_predict)

    return y_Predict


def prepare_data():
    # read csv from static folder
    pswd_data = pd.read_csv("Strength_Check/static/data.csv", error_bad_lines=False)
    # Removing missing values (preparing data)
    pswd_data.dropna(inplace=True)
    # converting dataset into numpy array
    pswd = np.array(pswd_data)
    # shuffling the dataset
    random.shuffle(pswd)
    return pswd


def main_function(check_password):
    pswd = prepare_data()
    # adding features and labels
    y_labels = [s[1] for s in pswd]  # strength column of dataset with 669639 rows
    all_passwords = [s[0] for s in pswd]  # passwords column of dataset with 669639 rows

    X_predict = [check_password]  # input password to analyse strength

    strength = tokenizing(X_predict, all_passwords, y_labels)

    """ Analyzing the password strength """
    specChar = False
    ucChar = False
    numChar = False
    pswd_len = len(check_password)
    list_pass = list(check_password)
    sequence = 0
    for i in range(1, len(list_pass)):
        if sequence >= 3:
            break
        elif abs(ord(list_pass[i]) - ord(list_pass[i - 1])) == 1:
            sequence += 1
    #  special character
    specialMatch = re.search(r'([^a-zA-Z0-9]+)', check_password, re.M)
    if specialMatch:
        specChar = True

    # Uppercase Character
    ucMatch = re.search(r'([A-Z])', check_password, re.M)
    if ucMatch:
        ucChar = True

    # Numeric Character
    numMatch = re.search(r'([0-9])', check_password, re.M)
    if numMatch:
        numChar = True

    if strength[0] == 0:

        if pswd_len < 8:
            if numChar == False or ucChar == False or specChar == False or sequence == 3:
                strenght_msg = "Your password is very weak"
                color= "red"
                msg1 = """Make sure 1. password length is atleast 8,
                       2. atleast one Captial letter
                       3. one number and one special character
                       4. Never have password characters in sequential order"""

                return (strenght_msg,color, msg1)
            else:
                strenght_msg = "Your password is very weak"
                color = "red"
                msg1 = "Make sure password length is atleast 8"
                return (strenght_msg,color, msg1)

        elif pswd_len > 7 and pswd_len < 13:
            if numChar == False or ucChar == False or specChar == False or sequence == 3:
                strenght_msg = "Your password is weak"
                color = "red"
                msg1 = """ Make sure your password has 
                       1. atleast one Captial letter,
                       2. one number and one special character,
                       3. Never have password characters in sequential order"""
                return (strenght_msg,color, msg1)
            else:
                strenght_msg = "Your password is weak"
                color = "red"
                msg1 = "Password appeared in leaked database"
                return (strenght_msg,color, msg1)
        else:
            strenght_msg = "Your password is weak"
            color = "red"
            msg1 = "Password appeared in leaked database"
            return (strenght_msg,color, msg1)

    if strength[0] == 1:
        if pswd_len < 8:
            if numChar == False or ucChar == False or specChar == False or sequence == 3:
                strenght_msg = "Your password has medium strength"
                color = "orange"
                msg1 = """Make sure 1. password length is atleast 8,
                       2. atleast one Captial letter,
                       3. one number and one special character,
                       4. Never have password characters in sequential order"""

                return (strenght_msg,color, msg1)
            else:
                strenght_msg = "Your password has medium strength"
                color = "orange"
                msg1 = "Make sure password length is atleast 8"
                return (strenght_msg,color, msg1)

        elif pswd_len > 7 and pswd_len < 17:
            if numChar == False or ucChar == False or specChar == False or sequence == 3:
                strenght_msg = "Your password has medium strength"
                color = "orange"
                msg1 = """ Make sure your password has 
                       1. atleast one Captial letter,
                       2. one number and one special character,
                       3. Never have password characters in sequential order"""
                return (strenght_msg,color, msg1)
            else:
                strenght_msg = "Your password has medium strength"
                color = "orange"
                msg1 = "Try my password generator for Stronger password! ;)"
                return (strenght_msg, color, msg1)
        else:
            strenght_msg = "Your password has medium strength"
            color = "orange"
            msg1 = "Try my password generator for Stronger password! ;)"
            return (strenght_msg,color, msg1)

    if strength[0] == 2:
        if pswd_len < 10:
            if numChar == False or ucChar == False or specChar == False or sequence == 3:
                strenght_msg = "Your password is strong"
                color = "green"
                msg1 = """Make sure 1. password length is atleast 10,
                       2. atleast one Captial letter,
                       3. one number and one special character,
                       4. Never have password characters in sequential order"""

                return (strenght_msg,color, msg1)
            else:
                strenght_msg = "Your password is strong"
                color = "green"
                msg1 = "Make sure password length is atleast 10"
                return (strenght_msg,color, msg1)

        elif pswd_len > 10 and pswd_len < 25:
            if numChar == False or ucChar == False or specChar == False or sequence == 3:
                strenght_msg = "Your password is strong"
                color = "green"
                msg1 = """ Make sure your password has 
                       1. atleast one Captial letter,
                       2. one number and one special character,
                       3. Never have password characters in sequential order"""
                return (strenght_msg,color, msg1)
            else:
                strenght_msg = "Your password is very strong"
                color = "green"
                msg1 = "Your data is safe!"
                return (strenght_msg,color, msg1)
        else:
            strenght_msg = "Your password is very strong"
            color = "green"
            msg1 = "Your data is safe!"
            return (strenght_msg,color, msg1)


# print(main_function('ABTYN123cg'))

"""logit = LogisticRegression(penalty = 'l2', multi_class= 'ovr')
logit.fit(X_train, y_train)
print("Accuracy: ", logit.score(X_test, y_test))
# Using Default Tokenizer
vectorizer = TfidfVectorizer()
# Store vectors into X variable as Our X allpasswords
X1 = vectorizer.fit_transform(allpasswords)
X_train, X_test, y_train, y_test = train_test_split(X1, ylabels, test_size=0.2, random_state=42)
# Model Building
# using logistic regression
# Multi_class for fast algorithm
logit = LogisticRegression(penalty='l2',multi_class='ovr')
logit.fit(X_train, y_train)
print("Accuracy :",logit.score(X_test, y_test))
data = data.dropna()
data["strength"] = data["strength"].map({0: "Weak", 
                                         1: "Medium",
                                         2: "Strong"})
                                         

 # clf = DecisionTreeClassifier()
    # clf.fit(X_train, y_train)
    # print("Accuracy :", clf.score(X_test, y_test))
    logit = LogisticRegression(penalty='l2', multi_class='ovr')
    logit.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)
    # print(y_pred)
    # print("Accuracy :", clf.score(X_test, y_test))
    X_predict = vectorizer.transform(X_predict)
    y_Predict = logit.predict(X_predict)  # clf.predict(X_predict)
    return y_Predict
    
    # pswd_data.head()
    # number of classification in strength column
    # pswd_data['strength'].unique()
    # Removing missing values (preparing data)
    # pswd_data.isna().sum()
    pswd_data.dropna(inplace=True)
    print("dropna", dt.datetime.now() - st)
    # pswd_data["strength"] = pswd_data["strength"].map({0: "Weak",1: "Medium", 2: "Strong"})
    # pswd_data.isnull().sum()
    # converting dataset into numpy array
"""
