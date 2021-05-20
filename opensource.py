import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
# For scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler


# For encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
'''

class

    원본

    스케일1
    스케일2
    스케일3
    스케일4

    인코딩1
    인코딩2


'''

#
# class EasyCombination:
#     def __init__(self, name, age, address):
#         self.hello = '안녕하세요.'
#         self.name = name
#         self.age = age
#         self.address = address
#
#     def greeting(self):
#         print('{0} 저는 {1}입니다.'.format(self.hello, self.name))

class EasyCombination:
    std_scaled_x_train = None
    std_scaled_x_test = None

    minMax_scaled_x_train = None
    minMax_scaled_x_test = None

    maxAbs_scaled_x_train = None
    maxAbs_scaled_x_test = None

    robust_scaled_x_train = None
    robust_scaled_x_test = None

    encoded_label = None
    encoded_onehot = None


    def __init__(self, dataset, x_train, x_test, items):
        self.dataset = dataset
        self.x_train = x_train
        self.x_test = x_test
        self.items = items

    def encode(self):
        # LabelEncoder를 객체로 생성한 후 , fit( ) 과 transform( ) 으로 label 인코딩 수행.
        encoder = LabelEncoder()
        encoder.fit(self.items)
        self.encoded_label = encoder.transform(self.items)

        labels = self.encoded_label.reshape(-1,1)

        oh_encoder = OneHotEncoder()
        oh_encoder.fit(labels)
        self.encoded_onehot = oh_encoder.transform(labels)

    def scale(self):
        standard_scaler = StandardScaler()
        standard_scaler.fit(self.x_train)
        self.std_scaled_x_train = standard_scaler.transform(self.x_train)
        self.std_scaled_x_test = standard_scaler.transform(self.x_test)

        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(self.x_train)
        self.minMax_scaled_x_train = minmax_scaler.transform(self.x_train)
        self.minMax_scaled_x_test = minmax_scaler.transform(self.x_test)

        maxabs_scaler = MaxAbsScaler()
        maxabs_scaler.fit(self.x_train)
        self.maxAbs_scaled_x_train = maxabs_scaler.transform(self.x_train)
        self.maxAbs_scaled_x_test = maxabs_scaler.transform(self.x_test)

        robust_scaler = RobustScaler()
        robust_scaler.fit(self.x_train)
        self.robust_scaled_x_train = robust_scaler.transform(self.x_train)
        self.robust_scaled_x_test = robust_scaler.transform(self.x_test)

    def findBest(self):
        best_result = 0


        return best_result


