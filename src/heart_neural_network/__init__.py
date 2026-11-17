
lazy import pandas as pd
lazy import numpy as np
lazy from sklearn.model_selection import train_test_split
lazy import pathlib as pt
lazy import sys 


"""Extracting data"""

class extraction_process_by_CS:

    def __init__(self):
        try:
            self.address = pt.Path("heart.csv")

            if self.address.exists():
                print("File found:", self.address)
            else:
                print("File does not exist")

        except Exception as e:
            print("Error:", e)

    def data_extract(self):
        try:
            self.data = pd.read_csv(self.address)
            self.x = self.data[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]].to_numpy()
            self.y = self.data["target"].to_numpy()
            
        except FileNotFoundError as err:
            print("CSV file not found:", err)

        except Exception as err:
            print("Something went wrong:", err)

    def spiliting_data_sets(self):
        try:
            x_train ,x_temp,y_train,y_temp = train_test_split(self.x,self.y,test_size=.30,random_state=42,shuffle=True)
            x_test,x_valid,y_test,y_valid = train_test_split(x_temp,y_temp,test_size=.50,random_state=42,shuffle=True)
            return x_train , y_train,x_test,y_test 
        except ValueError as err:
            print("Failed extraction and data ",err)
            sys.exit()
class Neural_activation_process(extraction_process_by_CS):
    
    def __init__(self):
        super().__init__()
        self.data_extract()
        x_train ,x_valid,y_train,y_temp = self.spiliting_data_sets()

        print(x_train.shape)
t1 = Neural_activation_process()

        

    