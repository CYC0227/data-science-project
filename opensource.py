import pandas as pd
import numpy as np 

# For scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

# For encoding
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class EasyCombination:
    #label
    std_scaled_x_train_label = None
    std_scaled_x_test_label = None

    minMax_scaled_x_train_label = None
    minMax_scaled_x_test_label = None

    maxAbs_scaled_x_train_label = None
    maxAbs_scaled_x_test_label = None

    robust_scaled_x_train_label = None
    robust_scaled_x_test_label = None

    #OneHot
    std_scaled_x_train_OH = None
    std_scaled_x_test_OH = None

    minMax_scaled_x_train_OH = None
    minMax_scaled_x_test_OH = None

    maxAbs_scaled_x_train_OH = None
    maxAbs_scaled_x_test_OH = None

    robust_scaled_x_train_OH = None
    robust_scaled_x_test_OH = None

    encoded_label = None
    encoded_onehot = None

    list_scale = []
    list_encode = []

    scores = []

    X_train_label = None
    X_test_label = None
    items = None

    dataset = None

    labelData = None
    ohData = None

    X_train_OH = None
    X_test_OH = None
    y_train_OH = None
    y_test_OH = None

    X_train_label = None
    X_test_label = None
    y_train_label = None
    y_test_label = None


    #accuracy scores
    score_label_std = None
    score_label_minMax = None
    score_label_maxAbs = None
    score_label_robust = None

    score_oh_std = None
    score_oh_minMax = None
    score_oh_maxAbs = None
    score_oh_robust = None

    results = []

    def __init__(self, dataset):
        self.dataset = dataset

    def encodeAndSplit(self, target_name):
        target_name
        target = [target_name]

        print(self.dataset.info())

        # Convert Categorical Value to Integer Value through Encoding
        # Label encoding usging a label encoder
        self.labelData = self.dataset
        self.ohData = self.dataset

        print("===============================Label encoding has started=============================================")
        
        label_encoder = LabelEncoder()
        for i in self.labelData.columns[self.labelData.dtypes == object]:
            print(i)
            self.labelData[i] = label_encoder.fit_transform(list(self.labelData[i]))

        print(self.labelData.head())
        print(self.labelData.info())

        print("===============================Label encoding is over=============================================")


        # One-Hot encoding using 'get_dummies' function
        print("===============================One-Hot encoding has started=============================================")

        for i in self.ohData.columns[self.ohData.dtypes == object]:
            print(i)
            self.ohData.drop(i, axis=1, inplace=True)
            self.ohData = pd.get_dummies(data=self.ohData, columns=[i], prefix=i)

        print(self.ohData.head())
        print(self.ohData.info())

        print("===============================One-Hot encoding is over=============================================")


        print("===============================Spliting has started=============================================")
        
        # Splitting the dataset into Training and Test Set
        # label
        X = self.labelData.drop(target, axis=1)
        y = self.labelData[target_name] 

        # The test set is allocated 30% of the total data set
        self.X_train_label, self.X_test_label, self.y_train_label, self.y_test_label = train_test_split(X, y,
                                                                                                        test_size=0.3,
                                                                                                        random_state=0)
        print(self.X_train_label.shape, self.y_train_label.shape)
        print(self.X_test_label.shape, self.y_test_label.shape)  

        # OneHot
        X = self.ohData.drop(target, axis=1)
        y = self.ohData[target_name] 

        # The test set is allocated 30% of the total data set
        self.X_train_OH, self.X_test_OH, self.y_train_OH, self.y_test_OH = train_test_split(X, y, test_size=0.3,
                                                                                            random_state=0)
        print(self.X_train_OH.shape, self.y_train_OH.shape)  
        print(self.X_test_OH.shape, self.y_test_OH.shape)  

        # Splitting the dataset into Training and Test Set

        # label
        X = self.labelData.drop([target_name], axis=1)
        y = self.labelData[target_name]

        # The test set is allocated 30% of the total data set
        self.X_train_label, self.X_test_label, self.y_train_label, self.y_test_label = train_test_split(X, y,
                                                                                                        test_size=0.3,
                                                                                                        random_state=0)
        print(self.X_train_label.shape, self.y_train_label.shape)  
        print(self.X_test_label.shape, self.y_test_label.shape) 

        # OneHot
        X = self.ohData.drop([target_name], axis=1)
        y = self.ohData[target_name]  

        # The test set is allocated 30% of the total data set
        self.X_train_OH, self.X_test_OH, self.y_train_OH, self.y_test_OH = train_test_split(X, y, test_size=0.3,
                                                                                            random_state=0)
        print(self.X_train_OH.shape, self.y_train_OH.shape)  # (51078, 123) (51078,)
        print(self.X_test_OH.shape, self.y_test_OH.shape)  # (21891, 123) (21891,)

        print("===============================Spliting is over=============================================")


    def scale(self):

        print("===============================Scaling has started=============================================")

        # label
        standard_scaler = StandardScaler()
        standard_scaler.fit(self.X_train_label)
        self.std_scaled_x_train_label = standard_scaler.transform(self.X_train_label)
        self.std_scaled_x_train_label = pd.DataFrame(self.std_scaled_x_train_label,
                                                     columns=self.X_train_label.columns)  # Converts back to dataframe after scaling
        self.std_scaled_x_test_label = standard_scaler.transform(self.X_test_label)
        self.std_scaled_x_test_label = pd.DataFrame(self.std_scaled_x_test_label,
                                                    columns=self.X_test_label.columns)  # Converts back to dataframe after scaling

        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(self.X_train_label)
        self.minMax_scaled_x_train_label = minmax_scaler.transform(self.X_train_label)
        self.minMax_scaled_x_train_label = pd.DataFrame(self.minMax_scaled_x_train_label,
                                                        columns=self.X_train_label.columns)  # Converts back to dataframe after scaling
        self.minMax_scaled_x_test_label = minmax_scaler.transform(self.X_test_label)
        self.minMax_scaled_x_test_label = pd.DataFrame(self.minMax_scaled_x_test_label,
                                                       columns=self.X_test_label.columns)  # Converts back to dataframe after scaling

        maxabs_scaler = MaxAbsScaler()
        maxabs_scaler.fit(self.X_train_label)
        self.maxAbs_scaled_x_train_label = maxabs_scaler.transform(self.X_train_label)
        self.maxAbs_scaled_x_train_label = pd.DataFrame(self.maxAbs_scaled_x_train_label,
                                                        columns=self.X_train_label.columns)  # Converts back to dataframe after scaling
        self.maxAbs_scaled_x_test_label = maxabs_scaler.transform(self.X_test_label)
        self.maxAbs_scaled_x_test_label = pd.DataFrame(self.maxAbs_scaled_x_test_label,
                                                       columns=self.X_test_label.columns)  # Converts back to dataframe after scaling

        robust_scaler = RobustScaler()
        robust_scaler.fit(self.X_train_label)
        self.robust_scaled_x_train_label = robust_scaler.transform(self.X_train_label)
        self.robust_scaled_x_train_label = pd.DataFrame(self.robust_scaled_x_train_label,
                                                        columns=self.X_train_label.columns)  # Converts back to dataframe after scaling
        self.robust_scaled_x_test_label = robust_scaler.transform(self.X_test_label)
        self.robust_scaled_x_test_label = pd.DataFrame(self.robust_scaled_x_test_label,
                                                       columns=self.X_test_label.columns)  # Converts back to dataframe after scaling


        # OneHot
        standard_scaler = StandardScaler()
        standard_scaler.fit(self.X_train_OH)
        self.std_scaled_x_train_OH = standard_scaler.transform(self.X_train_OH)
        self.std_scaled_x_train_OH = pd.DataFrame(self.std_scaled_x_train_OH,
                                                  columns=self.X_train_OH.columns)  # Converts back to dataframe after scaling
        self.std_scaled_x_test_OH = standard_scaler.transform(self.X_test_OH)
        self.std_scaled_x_test_OH = pd.DataFrame(self.std_scaled_x_test_OH,
                                                 columns=self.X_test_OH.columns)  # Converts back to dataframe after scaling

        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(self.X_train_OH)
        self.minMax_scaled_x_train_OH = minmax_scaler.transform(self.X_train_OH)
        self.minMax_scaled_x_train_OH = pd.DataFrame(self.minMax_scaled_x_train_OH,
                                                     columns=self.X_train_OH.columns)  # Converts back to dataframe after scaling
        self.minMax_scaled_x_test_OH = minmax_scaler.transform(self.X_test_OH)
        self.minMax_scaled_x_test_OH = pd.DataFrame(self.minMax_scaled_x_test_OH,
                                                    columns=self.X_test_OH.columns)  # Converts back to dataframe after scaling

        maxabs_scaler = MaxAbsScaler()
        maxabs_scaler.fit(self.X_train_OH)
        self.maxAbs_scaled_x_train_OH = maxabs_scaler.transform(self.X_train_OH)
        self.maxAbs_scaled_x_train_OH = pd.DataFrame(self.maxAbs_scaled_x_train_OH,
                                                     columns=self.X_train_OH.columns)  # Converts back to dataframe after scaling
        self.maxAbs_scaled_x_test_OH = maxabs_scaler.transform(self.X_test_OH)
        self.maxAbs_scaled_x_test_OH = pd.DataFrame(self.maxAbs_scaled_x_test_OH,
                                                    columns=self.X_test_OH.columns)  # Converts back to dataframe after scaling

        robust_scaler = RobustScaler()
        robust_scaler.fit(self.X_train_OH)
        self.robust_scaled_x_train_OH = robust_scaler.transform(self.X_train_OH)
        self.robust_scaled_x_train_OH = pd.DataFrame(self.robust_scaled_x_train_OH,
                                                     columns=self.X_train_OH.columns)  # Converts back to dataframe after scaling
        self.robust_scaled_x_test_OH = robust_scaler.transform(self.X_test_OH)
        self.robust_scaled_x_test_OH = pd.DataFrame(self.robust_scaled_x_test_OH,
                                                    columns=self.X_test_OH.columns)  # Converts back to dataframe after scaling

        print("===============================Scaling is over=============================================")

    def estimate(self, option):
        # option 1: DecisionTree
        # option 2: KNN
        # option 3: Random Forest
        # option others: Error
        
        if option == 1: #Decision Tree
            #Label Encoding + Standard Scaling
            print("\nSelected Decision Tree")
            DecisionTree = DecisionTreeClassifier(random_state=1)
            DecisionTree.fit(self.std_scaled_x_train_label, self.y_train_label)
            score_label_std = DecisionTree.score(self.std_scaled_x_test_label, self.y_test_label)
            self.score_label_std = score_label_std
            print('Score using Label Encoding and Standard Scaling:', score_label_std)

            #Label Encoding + MinMax Scaling
            DecisionTree = DecisionTreeClassifier(random_state=1)
            DecisionTree.fit(self.minMax_scaled_x_train_label, self.y_train_label)
            score_label_minMax = DecisionTree.score(self.minMax_scaled_x_test_label, self.y_test_label)
            self.score_label_minMax = score_label_minMax
            print('Score using Label Encoding and MinMax Scaling:', score_label_minMax)

            #Label Encoding + Maxabs Scaling
            DecisionTree = DecisionTreeClassifier(random_state=1)
            DecisionTree.fit(self.maxAbs_scaled_x_train_label, self.y_train_label)
            score_label_maxAbs = DecisionTree.score(self.maxAbs_scaled_x_test_label, self.y_test_label)
            self.score_label_maxAbs = score_label_maxAbs
            print('Score using Label Encoding and Maxabs Scaling:', score_label_maxAbs)
            
            #Label Encoding + Robust Scaling
            DecisionTree = DecisionTreeClassifier(random_state=1)
            DecisionTree.fit(self.robust_scaled_x_train_label, self.y_train_label)
            score_label_robust = DecisionTree.score(self.robust_scaled_x_test_label, self.y_test_label)
            self.score_label_robust = score_label_robust
            print('Score using Label Encoding and Robust Scaling:', score_label_robust)

            #One-Hot Encoding + Standard Scale
            DecisionTree = DecisionTreeClassifier(random_state=1)
            DecisionTree.fit(self.std_scaled_x_train_OH, self.y_train_OH)
            score_oh_std = DecisionTree.score(self.std_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_std = score_oh_std
            print('Score using One-Hot Encoding and Standard Scaling:', score_oh_std)

            #One-Hot Encoding + MinMax Scale
            DecisionTree = DecisionTreeClassifier(random_state=1)
            DecisionTree.fit(self.minMax_scaled_x_train_OH, self.y_train_OH)
            score_oh_minMax = DecisionTree.score(self.minMax_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_minMax = score_oh_minMax
            print('Score using One-Hot Encoding and MinMax Scaling:', score_oh_minMax)

            #One-Hot Encoding + Maxabs Scale
            DecisionTree = DecisionTreeClassifier(random_state=1)
            DecisionTree.fit(self.maxAbs_scaled_x_train_OH, self.y_train_OH)
            score_oh_maxAbs = DecisionTree.score(self.maxAbs_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_maxAbs = score_oh_maxAbs
            print('Score using One-Hot Encoding and Maxabs Scaling:', score_oh_maxAbs)
            
            #One-Hot Encoding + Robust Scale
            DecisionTree = DecisionTreeClassifier(random_state=1)
            DecisionTree.fit(self.robust_scaled_x_train_OH, self.y_train_OH)
            score_oh_robust = DecisionTree.score(self.robust_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_robust = score_oh_robust
            print('Score using One-Hot Encoding and Robust Scaling:', score_oh_robust)


        elif option == 2:  # KNN
            print("\nSelected K-nearest neighbors")
            # Label Encoding + Standard Scaling
            knn = KNeighborsClassifier()
            knn.fit(self.std_scaled_x_train_label, self.y_train_label)
            score_label_std = knn.score(self.std_scaled_x_test_label, self.y_test_label)
            self.score_label_std = score_label_std
            print('Score using Label Encoding and Standard Scaling:', score_label_std)

            # Label Encoding + MinMax Scaling
            knn = KNeighborsClassifier()
            knn.fit(self.minMax_scaled_x_train_label, self.y_train_label)
            score_label_minMax = knn.score(self.minMax_scaled_x_test_label, self.y_test_label)
            self.score_label_minMax = score_label_minMax
            print('Score using Label Encoding and MinMax Scaling:', score_label_minMax)

            # Label Encoding + Maxabs Scaling
            knn = KNeighborsClassifier()
            knn.fit(self.maxAbs_scaled_x_train_label, self.y_train_label)
            score_label_maxAbs = knn.score(self.maxAbs_scaled_x_test_label, self.y_test_label)
            self.score_label_maxAbs = score_label_maxAbs
            print('Score using Label Encoding and Maxabs Scaling:', score_label_maxAbs)

            # Label Encoding + Robust Scaling
            knn = KNeighborsClassifier()
            knn.fit(self.robust_scaled_x_train_label, self.y_train_label)
            score_label_robust = knn.score(self.robust_scaled_x_test_label, self.y_test_label)
            self.score_label_robust = score_label_robust
            print('Score using Label Encoding and Robust Scaling:', score_label_robust)

            # One-Hot Encoding + Standard Scale
            knn = KNeighborsClassifier()
            knn.fit(self.std_scaled_x_train_OH, self.y_train_OH)
            score_oh_std = knn.score(self.std_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_std = score_oh_std
            print('Score using One-Hot and Standard Scaling:', score_oh_std)

            # One-Hot Encoding + MinMax Scale
            knn = KNeighborsClassifier()
            knn.fit(self.minMax_scaled_x_train_OH, self.y_train_OH)
            score_oh_minMax = knn.score(self.minMax_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_minMax = score_oh_minMax
            print('Score using One-Hot Encoding and MinMax Scaling:', score_oh_minMax)

            # One-Hot Encoding + Maxabs Scale
            knn = KNeighborsClassifier()
            knn.fit(self.maxAbs_scaled_x_train_OH, self.y_train_OH)
            score_oh_maxAbs = knn.score(self.maxAbs_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_maxAbs = score_oh_maxAbs
            print('Score using One-Hot Encoding and Maxabs Scaling:', score_oh_maxAbs)

            # One-Hot Encoding + Robust Scale
            knn = KNeighborsClassifier()
            knn.fit(self.robust_scaled_x_train_OH, self.y_train_OH)
            score_oh_robust = knn.score(self.robust_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_robust = score_oh_robust
            print('Score using One-Hot Encoding and Robust Scaling:', score_oh_robust)


        elif option == 3: # Random Forest
            print("\nSelected Random Forest")
            #Label Encoding + Standard Scaling
            randomForest = RandomForestClassifier()
            randomForest.fit(self.std_scaled_x_train_label, self.y_train_label)
            score_label_std = randomForest.score(self.std_scaled_x_test_label, self.y_test_label)
            self.score_label_std = score_label_std
            print('Score using Label Encoding and Standard Scaling:', score_label_std)

            #Label Encoding + MinMax Scaling
            randomForest = RandomForestClassifier()
            randomForest.fit(self.minMax_scaled_x_train_label, self.y_train_label)
            score_label_minMax = randomForest.score(self.minMax_scaled_x_test_label, self.y_test_label)
            self.score_label_minMax = score_label_minMax
            print('Score using Label Encoding and MinMax Scaling:', score_label_minMax)

            #Label Encoding + Maxabs Scaling
            randomForest = RandomForestClassifier()
            randomForest.fit(self.maxAbs_scaled_x_train_label, self.y_train_label)
            score_label_maxAbs = randomForest.score(self.maxAbs_scaled_x_test_label, self.y_test_label)
            self.score_label_maxAbs = score_label_maxAbs
            print('Score using Label Encoding and Maxabs Scaling:', score_label_maxAbs)
            
            #Label Encoding + Robust Scaling
            randomForest = RandomForestClassifier()
            randomForest.fit(self.robust_scaled_x_train_label, self.y_train_label)
            score_label_robust = randomForest.score(self.robust_scaled_x_test_label, self.y_test_label)
            self.score_label_robust = score_label_robust
            print('Score using Label Encoding and Robust Scaling:', score_label_robust)

            #One-Hot Encoding + Standard Scale
            randomForest = RandomForestClassifier()
            randomForest.fit(self.std_scaled_x_train_OH, self.y_train_OH)
            score_oh_std = randomForest.score(self.std_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_std = score_oh_std
            print('Score using One-Hot and Standard Scaling:', score_oh_std)

            #One-Hot Encoding + MinMax Scale
            randomForest = RandomForestClassifier()
            randomForest.fit(self.minMax_scaled_x_train_OH, self.y_train_OH)
            score_oh_minMax = randomForest.score(self.minMax_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_minMax = score_oh_minMax
            print('Score using One-Hot Encoding and MinMax Scaling:', score_oh_minMax)

            #One-Hot Encoding + Maxabs Scale
            randomForest = RandomForestClassifier()
            randomForest.fit(self.maxAbs_scaled_x_train_OH, self.y_train_OH)
            score_oh_maxAbs = randomForest.score(self.maxAbs_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_maxAbs = score_oh_maxAbs
            print('Score using One-Hot Encoding and Maxabs Scaling:', score_oh_maxAbs)
            
            #One-Hot Encoding + Robust Scale
            randomForest = RandomForestClassifier()
            randomForest.fit(self.robust_scaled_x_train_OH, self.y_train_OH)
            score_oh_robust = randomForest.score(self.robust_scaled_x_test_OH, self.y_test_OH)
            self.score_oh_robust = score_oh_robust
            print('Score using One-Hot Encoding and Robust Scaling:', score_oh_robust)

        elif option != 1 or option != 2 or option != 3:
            print("ERROR: Wrong input! please try again...")
            return


    def findBestScore(self):
        a = self.score_label_std
        b = self.score_label_minMax
        c = self.score_label_maxAbs
        d = self.score_label_robust
        e = self.score_oh_std
        f = self.score_oh_minMax
        g = self.score_oh_maxAbs
        h = self.score_oh_robust
        
        results = [a, b, c, d, e, f, g, h]
        print("\nBest result score is:", max(results))

        if max(results) == a:
             print("Using Label Encoding and Standard Scaling")
        if max(results) == b:
            print("Using Label Encoding and MinMax Scaling")
        if max(results) == c:
            print("Using Label Encoding and Maxabs Scaling")
        if max(results) == d:
            print("Using Label Encoding and Robust Scaling")
        if max(results) == e:
            print("Using One-Hot Encoding and Standard Scaling")
        if max(results) == f:
            print("Using One-Hot Encoding and MinMax Scaling")
        if max(results) == g:
            print("Using One-Hot Encoding and Maxabs Scaling")
        if max(results) == h:
            print("Using One-Hot Encoding and Robust Scaling")


    def findWorstScore(self):
        a = self.score_label_std
        b = self.score_label_minMax
        c = self.score_label_maxAbs
        d = self.score_label_robust
        e = self.score_oh_std
        f = self.score_oh_minMax
        g = self.score_oh_maxAbs
        h = self.score_oh_robust

        results = [a, b, c, d, e, f, g, h]
        print("\nWorst result score is:", min(results))

        if min(results) == a:
             print("Using Label Encoding and Standard Scaling")
        if min(results) == b:
            print("Using Label Encoding and MinMax Scaling")
        if min(results) == c:
            print("Using Label Encoding and Maxabs Scaling")
        if min(results) == d:
            print("Using Label Encoding and Robust Scaling")
        if min(results) == e:
            print("Using One-Hot Encoding and Standard Scaling")
        if min(results) == f:
            print("Using One-Hot Encoding and MinMax Scaling")
        if min(results) == g:
            print("Using One-Hot Encoding and Maxabs Scaling")
        if min(results) == h:
            print("Using One-Hot Encoding and Robust Scaling")
        

    def printAllResult(self):
        
        print('\nPrint All results')
        a = self.score_label_std
        b = self.score_label_minMax
        c = self.score_label_maxAbs
        d = self.score_label_robust
        e = self.score_oh_std
        f = self.score_oh_minMax
        g = self.score_oh_maxAbs
        h = self.score_oh_robust

        results = [a, b, c, d, e, f, g, h]

        print("Score using Label Encoding and Standard Scaling:", a)
        print("Score using Label Encoding and MinMax Scaling:", b)
        print("Score using Label Encoding and Maxabs Scaling:", c)
        print("Score using Label Encoding and Robust Scaling:", d)
        print("Score using One-Hot Encoding and Standard Scaling:", e)
        print("Score using One-Hot Encoding and MinMax Scaling:", f)
        print("Score using One-Hot Encoding and Maxabs Scaling:", g)
        print("Score using One-Hot Encoding and Robust Scaling:", h)

        print("\nBest result score is:", max(results))
        if max(results) == a:
             print("Using Label Encoding and Standard Scaling")
        if max(results) == b:
            print("Using Label Encoding and MinMax Scaling")
        if max(results) == c:
            print("Using Label Encoding and Maxabs Scaling")
        if max(results) == d:
            print("Using Label Encoding and Robust Scaling")
        if max(results) == e:
            print("Using One-Hot Encoding and Standard Scaling")
        if max(results) == f:
            print("Using One-Hot Encoding and MinMax Scaling")
        if max(results) == g:
            print("Using One-Hot Encoding and Maxabs Scaling")
        if max(results) == h:
            print("Using One-Hot Encoding and Robust Scaling")

        print("\nWorst result score is:", min(results))
        if min(results) == a:
             print("Using Label Encoding and Standard Scaling")
        if min(results) == b:
            print("Using Label Encoding and MinMax Scaling")
        if min(results) == c:
            print("Using Label Encoding and Maxabs Scaling")
        if min(results) == d:
            print("Using Label Encoding and Robust Scaling")
        if min(results) == e:
            print("Using One-Hot Encoding and Standard Scaling")
        if min(results) == f:
            print("Using One-Hot Encoding and MinMax Scaling")
        if min(results) == g:
            print("Using One-Hot Encoding and Maxabs Scaling")
        if min(results) == h:
            print("Using One-Hot Encoding and Robust Scaling")

