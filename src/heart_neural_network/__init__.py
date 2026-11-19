
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

    def sigmoid(z):
        return 1/(1+(np.exp(-np.clip(z,-500,500))))

    @staticmethod
    def relu(z):return np.maximum(z*0.01,z)

    def relu_der(z):return np.where(z>0,1.0,0.01)

    def bce(pred, target):
        """Binary cross entropy: -[y*log(p) + (1-y)*log(1-p)]"""
        eps = 1e-15
        p = np.clip(pred, eps, 1.0 - eps)   
        loss = -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
        return np.mean(loss)
    
    def __init__(self):
        super().__init__()
        self.data_extract()
        x_train ,x_valid,y_train,y_temp = self.spiliting_data_sets() 

        w1 = np.random.randn(30,32)*np.sqrt(32/2)
        b1 = np.zeros((1,32))
        w2 = np.random.randn(32,16)*np.sqrt(16/2)
        b2 = np.zeros((1,24))
        w3 = np.random.randn(16,8)*np.sqrt(8/2)
        b3 = np.zeros((1,18))
        w4 = np.random.randn(8,4)*np.sqrt(4/2)
        b4 = np.zeros((1,4))
        w5 = np.random.randn(4,2)*np.sqrt(2/2)
        b5 = np.zeros((1,2))
        w6 = np.random.randn(2,1)*np.sqrt(1/2)
        b6 = np.zeros((1,1))
        lrt = 0.01
        ephocs = 10000 

        m = len(x_train)
        for ephoc in range(ephocs):
           indices = np.random.permutation(m)
           batch = 10
           x_shuffle ,y_shuffle = x_train[indices],y_train[indices]
           for i in range(0,m,batch):
            """Implementing the batch processing unit"""
            xb = x_shuffle[i:i+batch]
            yb = y_shuffle[i:i+batch]
            mb = len(xb)

            """ lets activate the Neuron 😎 """

            z1 = np.dot(xb,w1)+b1 
            a1 = self.relu(z1)

            z2 = np.dot(a1,w2)+b2 
            a2 = self.relu(z2)

            z3 = np.dot(a2,w3)+b3 
            a3 = self.relu(z3)

            z4 = np.dot(a3,w4)+b4 
            a4 = self.relu(z4)

            z5 = np.dot(a4,w5)+b5
            a5 = self.relu(z5)  

            z6 = np.dot(a5,w6)+b6
            a6 = self.sigmoid(z6) 

            """Backward propogation """   
            dz6 = (yb-y_shuffle)
            dw6 = (1/mb)*np.dot(a5.T,dz6)        
            db6 = (1/mb)*np.sum(dz6,axis=0,keepdims=True)

            dz5 = np.dot(dz6,w6.T)*self.relu_der(z5)
            dw5 = (1/mb)*np.dot(a4.T,dz5)
            db5 = (1/mb)*np.sum(dz5,axis=0,keepdims=True)

            dz4 = np.dot(dz5, w5.T) * self.relu_der(z4)
            dw4 = (1/mb)*np.dot(a3.T,dz4)
            db4 = (1/mb)*np.sum(dz4,axis=0,keepdims=True)

            dz3 = np.dot(dz4,w4.T)*self.relu_der(z3)
            dw3 = (1/mb)*np.dot(a2.T,dz3)
            db3 = (1/mb)*np.sum(dz3,axis=0,keepdims=True)

            dz2 = np.dot(dz3,w3.T) * self.relu_der(z2)
            dw2 = (1/mb)*np.dot(a1.T,dz2)
            db2 = (1/mb)*np.sum(dz2,axis=0,keepdims=True)

            dz1 = np.dot(dz2,w1.T) * self.relu_der(z1)
            dw1 = (1/mb)*np.dot(xb.T,dz1)
            db1 = (1/mb)*np.sum(dz1,axis=0,keepdims=True)

            w1 -= lrt * dw1
            w2 -= lrt * dw2
            w3 -= lrt * dw3    
            w4 -= lrt * dw4 
            w5 -= lrt * dw5
            w6 -= lrt * dw6 
            b1 -= lrt * db1   
            b2 -= lrt * db2 
            b3 -= lrt * db3 
            b4 -= lrt * db4  
            b5 -= lrt * db5 
            b6 -= lrt * db6 
        if ephoc%100==0:
            
           


t1 = Neural_activation_process()
     

    k