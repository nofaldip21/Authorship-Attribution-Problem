# Authorship-Attribution-Problem

# Introduction
Authorship attribution is the task of identifying the author of a given document. It has been studied extensively
in machine learning, natural language processing, linguistics, and privacy research and is widely applied in real-world
settings. Establishing the provenance of historical texts is helpful in placing important documents in their historical
contexts. In this assignment, we undertook a variety of machine learning methodologies to build a prolific authors
attribution prediction model in several publications. Given the information about authors, year, venue, title, and
abstract of publications, the machine learning model is designed to predict if the author of the publication is a prolific
author and which prolific author writes it.

# Evaluation Method
Every model and feature implementation in this report is evaluated by F1-score of the hold-out cross validation
to the training data. F1-score is a weighted average of precision and recall. Meanwhile hold-out cross validation is
splitting the training data into two parts. A part is responsible for training and the other part is used for testing.
