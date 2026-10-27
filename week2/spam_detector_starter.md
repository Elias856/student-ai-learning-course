# BRHA: lidt eksstra til seneste lektion

## Ekstra 2026-10-20

- Se mindst en video på https://www.youtube.com/playlist?list=PLE3ovFxnzNpblbg9cylEiOnctg0D24rKz
- Se gennem listen af videoer for at se om der kan være interessant at se noget på et senere tidspunkt
- Se også på listen af videoer her: https://www.youtube.com/@DeepLearningAI

# Week 2: Machine learning basics

## Meeting 3

Lesson 7: Supervised learning introduction
- Goal: Explain classification and regression 
- Activity: Use, understand and modify spam_detector_starter.py
- Activity: Use, understand and modify house_price_predictor.py
- Add more messages to the training data
- Try a different model from scikit-learn
- Look at https://pypi.org/project/scikit-learn/
- Look at https://scikit-learn.org/stable/

Lesson 8: Data preprocessing
- Goal: Teach cleaning and feature preparation
- Topics: missing values, text cleaning, train/test split
- Activity: Use, understand and modify preperation.py
  - The code teaches:
    - how to identify missing values
    - how to clean a dataset
    - how to convert categories to numbers
    - how to prepare features for machine learning
    - how to train a simple regression model
  - Modify some of these
    - Change the dataset
      - add more students
      - add new features like sleep_hours, homework_done, or motivation
    - Change the target
      - instead of final_score, predict exam_grade, hours_needed, or future_score
    - Try different missing value strategies
      - replace missing values with mean
      - replace with median
      - remove rows with missing values
    - Use a different feature encoding
      - map categories manually
      - use pandas.get_dummies()

Lesson 9: Intro to scikit-learn
- Goal: Show a complete ML workflow
- Activity: Train a logistic regression model

## Meeting 4

Lesson 10: Build a simple spam detector or sentiment model
- Goal: Use text data for classification
- Activity: Create a tiny dataset and train a model

Lesson 11: Model evaluation
- Goal: Explain accuracy and confusion matrix
- Activity: Discuss what bad accuracy means

Lesson 12: Code lab
- Goal: Students test their own ideas
- Activity: Tune parameters and compare results


# Week 2 Starter Code

This folder contains two simple machine learning example for Week 2.
spam_detector_starter.py
house_price_predictor.py

