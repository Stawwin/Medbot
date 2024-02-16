import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import os

# print("Current Working Directory:", os.getcwd())

def getDescription():
        global description_list
        with open('myapp\MasterData\symptom_Description.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                _description={row[0]:row[1]}
                description_list.update(_description)

def getSeverityDict():
        global severityDictionary
        with open('myapp\MasterData\Symptom_severity.csv') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            try:
                for row in csv_reader:
                    _diction={row[0]:int(row[1])}
                    severityDictionary.update(_diction)
            except:
                pass
            
def getprecautionDict():
        global precautionDictionary
        with open('myapp\MasterData\symptom_precaution.csv') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                _prec={row[0]:[row[1],row[2],row[3],row[4]]}
                precautionDictionary.update(_prec)

def getInfo():
        print("-----------------------------------Medikator-----------------------------------")
        print("\nYour Name? \t\t\t\t",end="->")
        # name=input("")
        # print("Hello, ",name)

def check_pattern(dis_list,inp):   #Function to predict the next symptom
        pred_list=[]                   #Array to store a list of the next predictions
        inp=inp.replace(' ','_')
        patt = f"{inp}"
        regexp = re.compile(patt)       #Check for matches using regular expressions
        pred_list=[item for item in dis_list if regexp.search(item)]    #adds matches into the prediction list
        if(len(pred_list)>0):
            return 1,pred_list
        else:
            return 0,[]                 #The list is returned until empty

def print_disease(node):
        node = node[0]
        val  = node.nonzero() 
        disease = le.inverse_transform(val[0])
        return list(map(lambda x:x.strip(),list(disease))) #Taking the leftmost element "x" from the CSV file, this is the predicted disease.

def tree_to_code(tree, feature_names, user_input):
        tree_ = tree.tree_                                 # Initializing data tree
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        chk_dis = ",".join(feature_names).split(",")
        symptoms_present = []

        # Now, instead of taking input from the console, use the user_input parameter
        disease_input = user_input
        conf, df_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("searches related to input: ")
            for num, it in enumerate(df_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = df_dis[conf_inp]
        else:
            print("Enter valid symptom.")

        while True:
            try:
                num_days = int(input("Okay. From how many days ? : "))
                break
            except:
                print("Enter valid input.")

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                if name == disease_input:
                    val = 1
                else:
                    val = 0
                if val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node])
                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                response = {"question": "Are you experiencing any", "symptoms": []}

                for syms in list(symptoms_given):
                    inp = ""
                    response["symptoms"].append({"symptom": syms, "question": f"{syms}?: "})

                # The following lines print the result. Modify as needed.
                if response["symptoms"]:
                    print(response["question"])
                    for sym in response["symptoms"]:
                        print(sym["question"], end="")
                        while True:
                            inp = input("")
                            if inp == "yes" or inp == "no":
                                break
                            else:
                                print("provide proper answers i.e. (yes/no) : ", end="")
                        if inp == "yes":
                            print("You are experiencing", sym["symptom"])
                        else:
                            print("You are not experiencing", sym["symptom"])
                else:
                    print("You may have ", present_disease[0])
                    print(description_list[present_disease[0]])

                precution_list = precautionDictionary[present_disease[0]]
                print("Take following measures : ")
                for i, j in enumerate(precution_list):
                    print(i + 1, ")", j)

def main():
    #Fetch the CSV file path
    csv_file_path_training = 'myapp\Data\Training.csv'
    csv_file_path_testing = 'myapp\Data\Testing.csv'

    #Load the files from the path
    training = pd.read_csv(csv_file_path_training)
    testing = pd.read_csv(csv_file_path_testing)

    #Define portions of CSV file
    cols= training.columns[:-1]
    cols= cols[:-1]
    x = training[cols]
    y = training['prognosis']
    y1= y

    # print("Sample Data:")
    # print(training.head())
    # print("\n")

    #Label encode the target variable
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)   #Test size takes a subset of the CSV file to train and test the bot. random_state ensures that the bot asks the same line of questioning each time.
    testx    = testing[cols]
    testy    = testing['prognosis']  
    testy    = le.transform(testy)


    #Train a Decision Tree model
    dt1 = DecisionTreeClassifier()
    dt = dt1.fit(x_train, y_train)

    #Train a SVM model
    model=SVC()
    model.fit(x_train, y_train)

    scores = cross_val_score(dt1, x_test, y_test, cv=3)
    # print (scores)
    print (scores.mean())

    print("for svm: ")
    print(model.score(x_test, y_test))

    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = cols

    severityDictionary=dict()
    description_list = dict()
    precautionDictionary=dict()

    symptoms_dict = {}

    

    reduced_data = training.groupby(training['prognosis']).max() #This will create a group of new symptoms each time a new symptom is entered.

    


    recurse(0, 1)
    getSeverityDict()
    getDescription()
    getprecautionDict()
    getInfo()
    tree_to_code(dt,cols, user_input="aga")
    print("----------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()