#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""
import sys
import pickle
sys.path.append("../final_project/")
from poi_email_addresses import poiEmails
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
emails = poiEmails()

totalSalary = 0
totalPoi = 0
# names = []
for name, features in enron_data.items():
    if features["poi"]:
       totalPoi += 1
    if features["poi"] and features["total_payments"] == "NaN":
        totalSalary += 1
print("Salary: " + str(totalSalary) + " " + str(totalPoi))
# print "Jeff"
# print enron_data["SKILLING JEFFREY K"]["total_payments"]
# print "Lay"
# print enron_data["LAY KENNETH L"]["total_payments"]
# print "Fastow"
# print enron_data["FASTOW ANDREW S"]["total_payments"]
