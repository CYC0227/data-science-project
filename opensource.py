import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sns as sns
from sklearn import preprocessing
# For scaling
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

# For encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier



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

    def __init__(self, dataset):
        self.dataset = dataset
        self.labelData = self.dataset
        self.ohData = self.dataset
        # self.x_train = x_train
        # self.x_test = x_test
        # self.items = items

    def encodeAndSplit(self):
        # Convert Categorical Value to Integer Value through Encoding
        # Label encoding usging a label encoder

        print("===============================Label start=============================================")
        label_encoder = LabelEncoder()
        for i in self.labelData.columns[self.labelData.dtypes == object]:
            print(i)
            self.labelData[i] = label_encoder.fit_transform(list(self.labelData[i]))
        print("===============================Label finish=============================================")

        # One-Hot encoding using 'get_dummies' function
        print("===============================OneHot start=============================================")
        print(self.ohData["Trim"].value_counts())
        print(self.ohData["Model"].value_counts())
        print(self.ohData["SubModel"].value_counts())

        # There are too many categories for one-hot encoding. Therefore, in this situation, the Trim, Model, and SubModel features are dropped
        self.ohData.drop(['Trim', 'Model', 'SubModel'], axis=1, inplace=True)

        for i in self.ohData.columns[self.ohData.dtypes == object]:
            print(i)
            self.ohData = pd.get_dummies(data=self.ohData, columns=[i], prefix=i)

        print(self.ohData.head())
        print(self.ohData.info())

        print("===============================OneHot finish=============================================")

        # Splitting the dataset into Training and Test Set

        # label
        X = self.labelData.drop(['IsBadBuy'], axis=1)
        y = self.labelData['IsBadBuy']  # IsBadBuy is the target feature

        # The test set is allocated 30% of the total data set
        self.X_train_label, self.X_test_label, self.y_train_label, self.y_test_label = train_test_split(X, y,
                                                                                                        test_size=0.3,
                                                                                                        random_state=0)
        print(self.X_train_label.shape, self.y_train_label.shape)  # (51078, 123) (51078,)
        print(self.X_test_label.shape, self.y_test_label.shape)  # (21891, 123) (21891,)

        # OneHot
        X = self.ohData.drop(['IsBadBuy'], axis=1)
        y = self.ohData['IsBadBuy']  # IsBadBuy is the target feature

        # The test set is allocated 30% of the total data set
        self.X_train_OH, self.X_test_OH, self.y_train_OH, self.y_test_OH = train_test_split(X, y, test_size=0.3,
                                                                                            random_state=0)
        print(self.X_train_OH.shape, self.y_train_OH.shape)  # (51078, 123) (51078,)
        print(self.X_test_OH.shape, self.y_test_OH.shape)  # (21891, 123) (21891,)

    def scale(self):
        # label

        standard_scaler = StandardScaler()
        standard_scaler.fit(self.X_train_label)
        self.std_scaled_x_train_label = standard_scaler.transform(self.X_train_label)
        self.std_scaled_x_train_label = pd.DataFrame(self.std_scaled_x_train_label,
                                                     columns=self.std_scaled_x_train_label.columns)  # Converts back to dataframe after scaling
        self.std_scaled_x_test_label = standard_scaler.transform(self.X_test_label)
        self.std_scaled_x_test_label = pd.DataFrame(self.std_scaled_x_test_label,
                                                    columns=self.std_scaled_x_test_label.columns)  # Converts back to dataframe after scaling

        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(self.X_train_label)
        self.minMax_scaled_x_train_label = minmax_scaler.transform(self.X_train_label)
        self.minMax_scaled_x_train_label = pd.DataFrame(self.minMax_scaled_x_train_label,
                                                        columns=self.minMax_scaled_x_train_label.columns)  # Converts back to dataframe after scaling
        self.minMax_scaled_x_test_label = minmax_scaler.transform(self.X_test_label)
        self.minMax_scaled_x_test_label = pd.DataFrame(self.minMax_scaled_x_test_label,
                                                       columns=self.minMax_scaled_x_test_label.columns)  # Converts back to dataframe after scaling

        maxabs_scaler = MaxAbsScaler()
        maxabs_scaler.fit(self.X_train_label)
        self.maxAbs_scaled_x_train_label = maxabs_scaler.transform(self.X_train_label)
        self.maxAbs_scaled_x_train_label = pd.DataFrame(self.maxAbs_scaled_x_train_label,
                                                        columns=self.maxAbs_scaled_x_train_label.columns)  # Converts back to dataframe after scaling
        self.maxAbs_scaled_x_test_label = maxabs_scaler.transform(self.X_test_label)
        self.maxAbs_scaled_x_test_label = pd.DataFrame(self.maxAbs_scaled_x_test_label,
                                                       columns=self.maxAbs_scaled_x_test_label.columns)  # Converts back to dataframe after scaling

        robust_scaler = RobustScaler()
        robust_scaler.fit(self.X_train_label)
        self.robust_scaled_x_train_label = robust_scaler.transform(self.X_train_label)
        self.robust_scaled_x_train_label = pd.DataFrame(self.robust_scaled_x_train_label,
                                                        columns=self.robust_scaled_x_train_label.columns)  # Converts back to dataframe after scaling
        self.robust_scaled_x_test_label = robust_scaler.transform(self.X_test_label)
        self.robust_scaled_x_test_label = pd.DataFrame(self.robust_scaled_x_test_label,
                                                       columns=self.robust_scaled_x_test_label.columns)  # Converts back to dataframe after scaling

        # OneHot
        standard_scaler = StandardScaler()
        standard_scaler.fit(self.X_train_label)
        self.std_scaled_x_train_OH = standard_scaler.transform(self.X_train_label)
        self.std_scaled_x_train_OH = pd.DataFrame(self.std_scaled_x_train_OH,
                                                  columns=self.std_scaled_x_train_OH.columns)  # Converts back to dataframe after scaling
        self.std_scaled_x_test_OH = standard_scaler.transform(self.X_test_label)
        self.std_scaled_x_test_OH = pd.DataFrame(self.std_scaled_x_test_OH,
                                                 columns=self.std_scaled_x_test_OH.columns)  # Converts back to dataframe after scaling

        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(self.X_train_label)
        self.minMax_scaled_x_train_OH = minmax_scaler.transform(self.X_train_label)
        self.minMax_scaled_x_train_OH = pd.DataFrame(self.minMax_scaled_x_train_OH,
                                                     columns=self.minMax_scaled_x_train_OH.columns)  # Converts back to dataframe after scaling
        self.minMax_scaled_x_test_OH = minmax_scaler.transform(self.X_test_label)
        self.minMax_scaled_x_test_OH = pd.DataFrame(self.minMax_scaled_x_test_OH,
                                                    columns=self.minMax_scaled_x_test_OH.columns)  # Converts back to dataframe after scaling

        maxabs_scaler = MaxAbsScaler()
        maxabs_scaler.fit(self.X_train_label)
        self.maxAbs_scaled_x_train_OH = maxabs_scaler.transform(self.X_train_label)
        self.maxAbs_scaled_x_train_OH = pd.DataFrame(self.maxAbs_scaled_x_train_OH,
                                                     columns=self.maxAbs_scaled_x_train_OH.columns)  # Converts back to dataframe after scaling
        self.maxAbs_scaled_x_test_OH = maxabs_scaler.transform(self.X_test_label)
        self.maxAbs_scaled_x_test_OH = pd.DataFrame(self.maxAbs_scaled_x_test_OH,
                                                    columns=self.maxAbs_scaled_x_test_OH.columns)  # Converts back to dataframe after scaling

        robust_scaler = RobustScaler()
        robust_scaler.fit(self.X_train_label)
        self.robust_scaled_x_train_OH = robust_scaler.transform(self.X_train_label)
        self.robust_scaled_x_train_OH = pd.DataFrame(self.robust_scaled_x_train_OH,
                                                     columns=self.robust_scaled_x_train_OH.columns)  # Converts back to dataframe after scaling
        self.robust_scaled_x_test_OH = robust_scaler.transform(self.X_test_label)
        self.robust_scaled_x_test_OH = pd.DataFrame(self.robust_scaled_x_test_OH,
                                                    columns=self.robust_scaled_x_test_OH.columns)  # Converts back to dataframe after scaling


    def estimate(self):
        import seaborn as sns
        #
        # #Label Encoding + Standard Scale
        #
        #
        #
        # params = {
        #     'max_depth': [6, 8, 10, 12, 16, 20, 24],
        #     'min_samples_split': [16, 24]
        # }
        # DecisionTree = DecisionTreeClassifier(random_state=156)
        # grid_cv = GridSearchCV(DecisionTree, param_grid=params, scoring='accuracy', cv=5, verbose=1)
        # grid_cv.fit(self.std_scaled_x_train_label, self.y_train_label)
        # print('Decision tree best score: {:.4f}'.format(grid_cv.best_score_))
        # print('Decision tree best parameter: ', grid_cv.best_params_)
        # # 최적의 파라미터값을 만든 모델
        # best_df_clf = grid_cv.best_estimator_
        # pred_dt = best_df_clf.predict(self.std_scaled_x_test_label)
        # # decision tree 예측결과
        # submission = pd.DataFrame(data=pred_dt, columns=["IsBadBuy"])
        # print("Decision tree best estimator prediction:")
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_dt)
        # print('Decision Tree best estimator accuracy: {0:.4f}'.format(accuracy))
        # # feature의 중요도 plt로 나타내기
        # feature_importance_values = best_df_clf.feature_importances_
        # # Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
        # feature_importances = pd.Series(feature_importance_values, index=self.X_train_label.columns)
        # # 중요도값 순으로 Series를 정렬
        # feature_top20 = feature_importances.sort_values(ascending=False)[:20]
        # feature_top1 = feature_top20.index[0]
        # feature_top2 = feature_top20.index[1]
        # print("feature top1: {0}, feature top2: {1}\n".format(feature_top1, feature_top2))
        # plt.figure(figsize=[8, 6])
        # plt.title('Feature Importances Top 20')
        # sns.barplot(x=feature_top20, y=feature_top20.index)
        # plt.show()
        #
        # # KNN algorithm evaluation
        # param_grid = {'n_neighbors': [1, 2, 3, 4, 5]}
        # from sklearn.neighbors import KNeighborsClassifier
        # estimator = KNeighborsClassifier()
        # grid = GridSearchCV(estimator, param_grid=param_grid)
        #
        # grid.fit(self.std_scaled_x_train_label, self.y_train_label)
        # print("K-nearest neighbors best score: {0:.4f}".format(grid.best_score_))
        # print("best hyperparameter:")
        # print(grid.best_params_)
        # # KNN 최적의 모델
        # best_df_knn = grid.best_estimator_
        # # 최적 모델 예측결과
        # pred_knn = best_df_knn.predict(self.std_scaled_x_test_label)
        # submission = pd.DataFrame(data=pred_knn, columns=["IsBadBuy"])
        # print("best KNN predict:")
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_knn)
        # print('k-nearest neighbor accuracy: {0:.4f}'.format(accuracy))
        # print("\n")
        #
        # # ensemble learning - Random Forest
        # # instantiate model with 1000 decision trees
        # from sklearn.ensemble import RandomForestClassifier
        # rf = RandomForestClassifier()
        # rf_param_grid = {
        #     'n_estimators': [100, 200],
        #     'max_depth': [6, 8, 10, 12],
        #     'min_samples_leaf': [3, 5, 7, 10],
        #     'min_samples_split': [2, 3, 5, 10]
        # }
        # rf_grid = GridSearchCV(rf, param_grid=rf_param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
        # rf_grid.fit(self.std_scaled_x_train_label, self.y_train_label)
        # print('Random forest best score: {:.4f}'.format(rf_grid.best_score_))
        # print('Random forest best parameter: ', rf_grid.best_params_)
        # # 최적의 파라미터값을 만든 모델
        # best_rf_clf = rf_grid.best_estimator_
        # pred_rf = best_rf_clf.predict(self.std_scaled_x_test_label)
        # # Random Forest 예측결과
        # print("Random Forest best estimator prediction")
        # submission = pd.DataFrame(data=pred_rf, columns=["IsBadBuy"])
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_rf)
        # print('Random Forest accuracy: {0:.4f}'.format(accuracy))
        # print("\n")
        #
        #
        #
        #
        #
        #
        #
        # #Label Encoding + MinMax Scale
        #
        # params = {
        #     'max_depth': [6, 8, 10, 12, 16, 20, 24],
        #     'min_samples_split': [16, 24]
        # }
        # DecisionTree = DecisionTreeClassifier(random_state=156)
        # grid_cv = GridSearchCV(DecisionTree, param_grid=params, scoring='accuracy', cv=5, verbose=1)
        # grid_cv.fit(self.minMax_scaled_x_train_label, self.y_train_OH)
        # print('Decision tree best score: {:.4f}'.format(grid_cv.best_score_))
        # print('Decision tree best parameter: ', grid_cv.best_params_)
        #
        # ('Decision tree best parameter: ', grid_cv.best_params_)
        # # 최적의 파라미터값을 만든 모델
        # best_df_clf = grid_cv.best_estimator_
        # pred_dt = best_df_clf.predict(self.minMax_scaled_x_test_label)
        # # decision tree 예측결과
        # submission = pd.DataFrame(data=pred_dt, columns=["IsBadBuy"])
        # print("Decision tree best estimator prediction:")
        # print(submission)
        #
        #
        # accuracy = accuracy_score(self.y_test_label, pred_dt)
        # print('Decision Tree best estimator accuracy: {0:.4f}'.format(accuracy))
        # # feature의 중요도 plt로 나타내기
        # feature_importance_values = best_df_clf.feature_importances_
        # # Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
        # feature_importances = pd.Series(feature_importance_values, index=self.X_train_label.columns)
        # # 중요도값 순으로 Series를 정렬
        # feature_top20 = feature_importances.sort_values(ascending=False)[:20]
        # feature_top1 = feature_top20.index[0]
        # feature_top2 = feature_top20.index[1]
        # print("feature top1: {0}, feature top2: {1}\n".format(feature_top1, feature_top2))
        # plt.figure(figsize=[8, 6])
        # plt.title('Feature Importances Top 20')
        # sns.barplot(x=feature_top20, y=feature_top20.index)
        # plt.show()
        #
        # # KNN algorithm evaluation
        # from sklearn.neighbors import KNeighborsClassifier
        # param_grid = {'n_neighbors': [1, 2, 3, 4, 5]}
        # estimator = KNeighborsClassifier()
        # grid = GridSearchCV(estimator, param_grid=param_grid)
        #
        #
        # grid.fit(self.minMax_scaled_x_train_label, self.y_train_label)
        # print("K-nearest neighbors best score: {0:.4f}".format(grid.best_score_))
        # print("best hyperparameter:")
        # print(grid.best_params_)
        # # KNN 최적의 모델
        # best_df_knn = grid.best_estimator_
        # # 최적 모델 예측결과
        # pred_knn = best_df_knn.predict(self.minMax_scaled_x_test_label)
        # submission = pd.DataFrame(data=pred_knn, columns=["IsBadBuy"])
        # print("best KNN predict:")
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_knn)
        # print('k-nearest neighbor accuracy: {0:.4f}'.format(accuracy))
        # print("\n")
        #
        # # ensemble learning - Random Forest
        # from sklearn.ensemble import RandomForestClassifier
        # # instantiate model with 1000 decision trees
        # rf = RandomForestClassifier()
        # rf_param_grid = {
        #     'n_estimators': [100, 200],
        #     'max_depth': [6, 8, 10, 12],
        #     'min_samples_leaf': [3, 5, 7, 10],
        #     'min_samples_split': [2, 3, 5, 10]
        # }
        # rf_grid = GridSearchCV(rf, param_grid=rf_param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
        # rf_grid.fit(self.minMax_scaled_x_train_label, self.y_train_label)
        # print('Random forest best score: {:.4f}'.format(rf_grid.best_score_))
        # print('Random forest best parameter: ', rf_grid.best_params_)
        # # 최적의 파라미터값을 만든 모델
        # best_rf_clf = rf_grid.best_estimator_
        # pred_rf = best_rf_clf.predict(self.minMax_scaled_x_test_label)
        # # Random Forest 예측결과
        # print("Random Forest best estimator prediction")
        # submission = pd.DataFrame(data=pred_rf, columns=["IsBadBuy"])
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_rf)
        # print('Random Forest accuracy: {0:.4f}'.format(accuracy))
        # print("\n")
        #
        #
        #
        #
        #
        #
        #
        #
        # #Label Encoding + MaxAbs Scale
        #
        # params = {
        #     'max_depth': [6, 8, 10, 12, 16, 20, 24],
        #     'min_samples_split': [16, 24]
        # }
        # DecisionTree = DecisionTreeClassifier(random_state=156)
        # grid_cv = GridSearchCV(DecisionTree, param_grid=params, scoring='accuracy', cv=5, verbose=1)
        # grid_cv.fit(self.maxAbs_scaled_x_train_label, self.y_train_label)
        # print('Decision tree best score: {:.4f}'.format(grid_cv.best_score_))
        # print('Decision tree best parameter: ', grid_cv.best_params_)
        # # 최적의 파라미터값을 만든 모델
        # best_df_clf = grid_cv.best_estimator_
        # pred_dt = best_df_clf.predict(self.maxAbs_scaled_x_test_label)
        # # decision tree 예측결과
        # submission = pd.DataFrame(data=pred_dt, columns=["IsBadBuy"])
        # print("Decision tree best estimator prediction:")
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_dt)
        # print('Decision Tree best estimator accuracy: {0:.4f}'.format(accuracy))
        # # feature의 중요도 plt로 나타내기
        # feature_importance_values = best_df_clf.feature_importances_
        # # Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
        # feature_importances = pd.Series(feature_importance_values, index=self.X_train_label.columns)
        # # 중요도값 순으로 Series를 정렬
        # feature_top20 = feature_importances.sort_values(ascending=False)[:20]
        # feature_top1 = feature_top20.index[0]
        # feature_top2 = feature_top20.index[1]
        # print("feature top1: {0}, feature top2: {1}\n".format(feature_top1, feature_top2))
        # plt.figure(figsize=[8, 6])
        # plt.title('Feature Importances Top 20')
        # sns.barplot(x=feature_top20, y=feature_top20.index)
        # plt.show()
        #
        # # KNN algorithm evaluation
        # param_grid = {'n_neighbors': [1, 2, 3, 4, 5]}
        # estimator = KNeighborsClassifier()
        # grid = GridSearchCV(estimator, param_grid=param_grid)
        #
        # grid.fit(self.maxAbs_scaled_x_train_label, self.y_train_label)
        # print("K-nearest neighbors best score: {0:.4f}".format(grid.best_score_))
        # print("best hyperparameter:")
        # print(grid.best_params_)
        # # KNN 최적의 모델
        # best_df_knn = grid.best_estimator_
        # # 최적 모델 예측결과
        # pred_knn = best_df_knn.predict(self.maxAbs_scaled_x_test_label)
        # submission = pd.DataFrame(data=pred_knn, columns=["IsBadBuy"])
        # print("best KNN predict:")
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_knn)
        # print('k-nearest neighbor accuracy: {0:.4f}'.format(accuracy))
        # print("\n")
        #
        # # ensemble learning - Random Forest
        # # instantiate model with 1000 decision trees
        # rf = RandomForestClassifier()
        # rf_param_grid = {
        #     'n_estimators': [100, 200],
        #     'max_depth': [6, 8, 10, 12],
        #     'min_samples_leaf': [3, 5, 7, 10],
        #     'min_samples_split': [2, 3, 5, 10]
        # }
        # rf_grid = GridSearchCV(rf, param_grid=rf_param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
        # rf_grid.fit(self.maxAbs_scaled_x_train_label, self.y_train_label)
        # print('Random forest best score: {:.4f}'.format(rf_grid.best_score_))
        # print('Random forest best parameter: ', rf_grid.best_params_)
        # # 최적의 파라미터값을 만든 모델
        # best_rf_clf = rf_grid.best_estimator_
        # pred_rf = best_rf_clf.predict(self.maxAbs_scaled_x_test_label)
        # # Random Forest 예측결과
        # print("Random Forest best estimator prediction")
        # submission = pd.DataFrame(data=pred_rf, columns=["IsBadBuy"])
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_rf)
        # print('Random Forest accuracy: {0:.4f}'.format(accuracy))
        # print("\n")
        #
        #
        #
        #
        #
        #
        # #Label Encoding + Robust Scale
        #
        #
        # params = {
        #     'max_depth': [6, 8, 10, 12, 16, 20, 24],
        #     'min_samples_split': [16, 24]
        # }
        # DecisionTree = DecisionTreeClassifier(random_state=156)
        # grid_cv = GridSearchCV(DecisionTree, param_grid=params, scoring='accuracy', cv=5, verbose=1)
        # grid_cv.fit(self.robust_scaled_x_train_label, self.y_train_label)
        # print('Decision tree best score: {:.4f}'.format(grid_cv.best_score_))
        # print('Decision tree best parameter: ', grid_cv.best_params_)
        # # 최적의 파라미터값을 만든 모델
        # best_df_clf = grid_cv.best_estimator_
        # pred_dt = best_df_clf.predict(self.robust_scaled_x_test_label)
        # # decision tree 예측결과
        # submission = pd.DataFrame(data=pred_dt, columns=["IsBadBuy"])
        # print("Decision tree best estimator prediction:")
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_dt)
        # print('Decision Tree best estimator accuracy: {0:.4f}'.format(accuracy))
        # # feature의 중요도 plt로 나타내기
        # feature_importance_values = best_df_clf.feature_importances_
        # # Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
        # feature_importances = pd.Series(feature_importance_values, index=self.X_train_label.columns)
        # # 중요도값 순으로 Series를 정렬
        # feature_top20 = feature_importances.sort_values(ascending=False)[:20]
        # feature_top1 = feature_top20.index[0]
        # feature_top2 = feature_top20.index[1]
        # print("feature top1: {0}, feature top2: {1}\n".format(feature_top1, feature_top2))
        # plt.figure(figsize=[8, 6])
        # plt.title('Feature Importances Top 20')
        # sns.barplot(x=feature_top20, y=feature_top20.index)
        # plt.show()
        #
        # # KNN algorithm evaluation
        # param_grid = {'n_neighbors': [1, 2, 3, 4, 5]}
        # estimator = KNeighborsClassifier()
        # grid = GridSearchCV(estimator, param_grid=param_grid)
        #
        # grid.fit(self.robust_scaled_x_train_label, self.y_train_label)
        # print("K-nearest neighbors best score: {0:.4f}".format(grid.best_score_))
        # print("best hyperparameter:")
        # print(grid.best_params_)
        # # KNN 최적의 모델
        # best_df_knn = grid.best_estimator_
        # # 최적 모델 예측결과
        # pred_knn = best_df_knn.predict(self.robust_scaled_x_test_label)
        # submission = pd.DataFrame(data=pred_knn, columns=["IsBadBuy"])
        # print("best KNN predict:")
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_knn)
        # print('k-nearest neighbor accuracy: {0:.4f}'.format(accuracy))
        # print("\n")
        #
        # # ensemble learning - Random Forest
        # # instantiate model with 1000 decision trees
        # rf = RandomForestClassifier()
        # rf_param_grid = {
        #     'n_estimators': [100, 200],
        #     'max_depth': [6, 8, 10, 12],
        #     'min_samples_leaf': [3, 5, 7, 10],
        #     'min_samples_split': [2, 3, 5, 10]
        # }
        # rf_grid = GridSearchCV(rf, param_grid=rf_param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
        # rf_grid.fit(self.robust_scaled_x_train_label, self.y_train_label)
        # print('Random forest best score: {:.4f}'.format(rf_grid.best_score_))
        # print('Random forest best parameter: ', rf_grid.best_params_)
        # # 최적의 파라미터값을 만든 모델
        # best_rf_clf = rf_grid.best_estimator_
        # pred_rf = best_rf_clf.predict(self.robust_scaled_x_test_label)
        # # Random Forest 예측결과
        # print("Random Forest best estimator prediction")
        # submission = pd.DataFrame(data=pred_rf, columns=["IsBadBuy"])
        # print(submission)
        #
        # accuracy = accuracy_score(self.y_test_label, pred_rf)
        # print('Random Forest accuracy: {0:.4f}'.format(accuracy))
        # print("\n")
        #
        #
        #





        # OneHot Encoding + Standard Scale

        params = {
            'max_depth': [6, 8, 10, 12, 16, 20, 24],
            'min_samples_split': [16, 24]
        }
        DecisionTree = DecisionTreeClassifier(random_state=156)
        grid_cv = GridSearchCV(DecisionTree, param_grid=params, scoring='accuracy', cv=5, verbose=1)
        grid_cv.fit(self.std_scaled_x_train_OH , self.y_train_OH)
        print('Decision tree best score: {:.4f}'.format(grid_cv.best_score_))
        print('Decision tree best parameter: ', grid_cv.best_params_)
        # 최적의 파라미터값을 만든 모델
        best_df_clf = grid_cv.best_estimator_
        pred_dt = best_df_clf.predict(self.std_scaled_x_test_OH )
        # decision tree 예측결과
        submission = pd.DataFrame(data=pred_dt, columns=["IsBadBuy"])
        print("Decision tree best estimator prediction:")
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_dt)
        print('Decision Tree best estimator accuracy: {0:.4f}'.format(accuracy))
        # feature의 중요도 plt로 나타내기
        import seaborn as sns
        feature_importance_values = best_df_clf.feature_importances_
        # Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
        feature_importances = pd.Series(feature_importance_values, index=self.y_train_OH.columns)
        # 중요도값 순으로 Series를 정렬
        feature_top20 = feature_importances.sort_values(ascending=False)[:20]
        feature_top1 = feature_top20.index[0]
        feature_top2 = feature_top20.index[1]
        print("feature top1: {0}, feature top2: {1}\n".format(feature_top1, feature_top2))
        plt.figure(figsize=[8, 6])
        plt.title('Feature Importances Top 20')
        sns.barplot(x=feature_top20, y=feature_top20.index)
        plt.show()

        # KNN algorithm evaluation
        from sklearn.neighbors import KNeighborsClassifier
        param_grid = {'n_neighbors': [1, 2, 3, 4, 5]}
        estimator = KNeighborsClassifier()
        grid = GridSearchCV(estimator, param_grid=param_grid)

        grid.fit(self.std_scaled_x_train_OH , self.y_train_OH)
        print("K-nearest neighbors best score: {0:.4f}".format(grid.best_score_))
        print("best hyperparameter:")
        print(grid.best_params_)
        # KNN 최적의 모델
        best_df_knn = grid.best_estimator_
        # 최적 모델 예측결과
        pred_knn = best_df_knn.predict(self.std_scaled_x_test_OH )
        submission = pd.DataFrame(data=pred_knn, columns=["IsBadBuy"])
        print("best KNN predict:")
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_knn)
        print('k-nearest neighbor accuracy: {0:.4f}'.format(accuracy))
        print("\n")

        # ensemble learning - Random Forest
        from sklearn.ensemble import RandomForestClassifier
        # instantiate model with 1000 decision trees
        rf = RandomForestClassifier()
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [3, 5, 7, 10],
            'min_samples_split': [2, 3, 5, 10]
        }
        rf_grid = GridSearchCV(rf, param_grid=rf_param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
        rf_grid.fit(self.std_scaled_x_train_OH , self.y_train_OH)
        print('Random forest best score: {:.4f}'.format(rf_grid.best_score_))
        print('Random forest best parameter: ', rf_grid.best_params_)
        # 최적의 파라미터값을 만든 모델
        best_rf_clf = rf_grid.best_estimator_
        pred_rf = best_rf_clf.predict(self.std_scaled_x_test_OH )
        # Random Forest 예측결과
        print("Random Forest best estimator prediction")
        submission = pd.DataFrame(data=pred_rf, columns=["IsBadBuy"])
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_rf)
        print('Random Forest accuracy: {0:.4f}'.format(accuracy))
        print("\n")








        # OneHot Encoding + MinMax Scale

        params = {
            'max_depth': [6, 8, 10, 12, 16, 20, 24],
            'min_samples_split': [16, 24]
        }
        DecisionTree = DecisionTreeClassifier(random_state=156)
        grid_cv = GridSearchCV(DecisionTree, param_grid=params, scoring='accuracy', cv=5, verbose=1)
        grid_cv.fit(self.minMax_scaled_x_train_OH , self.y_train_OH)
        print('Decision tree best score: {:.4f}'.format(grid_cv.best_score_))
        print('Decision tree best parameter: ', grid_cv.best_params_)
        # 최적의 파라미터값을 만든 모델
        best_df_clf = grid_cv.best_estimator_
        pred_dt = best_df_clf.predict(self.minMax_scaled_x_test_OH )
        # decision tree 예측결과
        submission = pd.DataFrame(data=pred_dt, columns=["IsBadBuy"])
        print("Decision tree best estimator prediction:")
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_dt)
        print('Decision Tree best estimator accuracy: {0:.4f}'.format(accuracy))
        # feature의 중요도 plt로 나타내기
        import seaborn as sns
        feature_importance_values = best_df_clf.feature_importances_
        # Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
        feature_importances = pd.Series(feature_importance_values, index=self.y_train_OH.columns)
        # 중요도값 순으로 Series를 정렬
        feature_top20 = feature_importances.sort_values(ascending=False)[:20]
        feature_top1 = feature_top20.index[0]
        feature_top2 = feature_top20.index[1]
        print("feature top1: {0}, feature top2: {1}\n".format(feature_top1, feature_top2))
        plt.figure(figsize=[8, 6])
        plt.title('Feature Importances Top 20')
        sns.barplot(x=feature_top20, y=feature_top20.index)
        plt.show()

        # KNN algorithm evaluation
        from sklearn.neighbors import KNeighborsClassifier
        param_grid = {'n_neighbors': [1, 2, 3, 4, 5]}
        estimator = KNeighborsClassifier()
        grid = GridSearchCV(estimator, param_grid=param_grid)

        grid.fit(self.minMax_scaled_x_train_OH , self.y_train_OH)
        print("K-nearest neighbors best score: {0:.4f}".format(grid.best_score_))
        print("best hyperparameter:")
        print(grid.best_params_)
        # KNN 최적의 모델
        best_df_knn = grid.best_estimator_
        # 최적 모델 예측결과
        pred_knn = best_df_knn.predict(self.minMax_scaled_x_test_OH )
        submission = pd.DataFrame(data=pred_knn, columns=["IsBadBuy"])
        print("best KNN predict:")
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_knn)
        print('k-nearest neighbor accuracy: {0:.4f}'.format(accuracy))
        print("\n")

        # ensemble learning - Random Forest
        from sklearn.ensemble import RandomForestClassifier
        # instantiate model with 1000 decision trees
        rf = RandomForestClassifier()
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [3, 5, 7, 10],
            'min_samples_split': [2, 3, 5, 10]
        }
        rf_grid = GridSearchCV(rf, param_grid=rf_param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
        rf_grid.fit(self.minMax_scaled_x_train_OH , self.y_train_OH)
        print('Random forest best score: {:.4f}'.format(rf_grid.best_score_))
        print('Random forest best parameter: ', rf_grid.best_params_)
        # 최적의 파라미터값을 만든 모델
        best_rf_clf = rf_grid.best_estimator_
        pred_rf = best_rf_clf.predict(self.minMax_scaled_x_test_OH )
        # Random Forest 예측결과
        print("Random Forest best estimator prediction")
        submission = pd.DataFrame(data=pred_rf, columns=["IsBadBuy"])
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_rf)
        print('Random Forest accuracy: {0:.4f}'.format(accuracy))
        print("\n")





        # OneHot Encoding + MaxAbs Scale

        params = {
            'max_depth': [6, 8, 10, 12, 16, 20, 24],
            'min_samples_split': [16, 24]
        }
        DecisionTree = DecisionTreeClassifier(random_state=156)
        grid_cv = GridSearchCV(DecisionTree, param_grid=params, scoring='accuracy', cv=5, verbose=1)
        grid_cv.fit(self.maxAbs_scaled_x_train_OH , self.y_train_OH)
        print('Decision tree best score: {:.4f}'.format(grid_cv.best_score_))
        print('Decision tree best parameter: ', grid_cv.best_params_)
        # 최적의 파라미터값을 만든 모델
        best_df_clf = grid_cv.best_estimator_
        pred_dt = best_df_clf.predict(self.maxAbs_scaled_x_test_OH )
        # decision tree 예측결과
        submission = pd.DataFrame(data=pred_dt, columns=["IsBadBuy"])
        print("Decision tree best estimator prediction:")
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_dt)
        print('Decision Tree best estimator accuracy: {0:.4f}'.format(accuracy))
        # feature의 중요도 plt로 나타내기
        import seaborn as sns
        feature_importance_values = best_df_clf.feature_importances_
        # Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
        feature_importances = pd.Series(feature_importance_values, index=self.y_train_OH.columns)
        # 중요도값 순으로 Series를 정렬
        feature_top20 = feature_importances.sort_values(ascending=False)[:20]
        feature_top1 = feature_top20.index[0]
        feature_top2 = feature_top20.index[1]
        print("feature top1: {0}, feature top2: {1}\n".format(feature_top1, feature_top2))
        plt.figure(figsize=[8, 6])
        plt.title('Feature Importances Top 20')
        sns.barplot(x=feature_top20, y=feature_top20.index)
        plt.show()

        # KNN algorithm evaluation
        from sklearn.neighbors import KNeighborsClassifier
        param_grid = {'n_neighbors': [1, 2, 3, 4, 5]}
        estimator = KNeighborsClassifier()
        grid = GridSearchCV(estimator, param_grid=param_grid)

        grid.fit(self.maxAbs_scaled_x_train_OH , self.y_train_OH)
        print("K-nearest neighbors best score: {0:.4f}".format(grid.best_score_))
        print("best hyperparameter:")
        print(grid.best_params_)
        # KNN 최적의 모델
        best_df_knn = grid.best_estimator_
        # 최적 모델 예측결과
        pred_knn = best_df_knn.predict(self.maxAbs_scaled_x_test_OH )
        submission = pd.DataFrame(data=pred_knn, columns=["IsBadBuy"])
        print("best KNN predict:")
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_knn)
        print('k-nearest neighbor accuracy: {0:.4f}'.format(accuracy))
        print("\n")

        # ensemble learning - Random Forest
        from sklearn.ensemble import RandomForestClassifier
        # instantiate model with 1000 decision trees
        rf = RandomForestClassifier()
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [3, 5, 7, 10],
            'min_samples_split': [2, 3, 5, 10]
        }
        rf_grid = GridSearchCV(rf, param_grid=rf_param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
        rf_grid.fit(self.maxAbs_scaled_x_train_OH , self.y_train_OH)
        print('Random forest best score: {:.4f}'.format(rf_grid.best_score_))
        print('Random forest best parameter: ', rf_grid.best_params_)
        # 최적의 파라미터값을 만든 모델
        best_rf_clf = rf_grid.best_estimator_
        pred_rf = best_rf_clf.predict(self.maxAbs_scaled_x_test_OH )
        # Random Forest 예측결과
        print("Random Forest best estimator prediction")
        submission = pd.DataFrame(data=pred_rf, columns=["IsBadBuy"])
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_rf)
        print('Random Forest accuracy: {0:.4f}'.format(accuracy))
        print("\n")





        # OneHot Encoding + Robust Scale

        params = {
            'max_depth': [6, 8, 10, 12, 16, 20, 24],
            'min_samples_split': [16, 24]
        }
        DecisionTree = DecisionTreeClassifier(random_state=156)
        grid_cv = GridSearchCV(DecisionTree, param_grid=params, scoring='accuracy', cv=5, verbose=1)
        grid_cv.fit(self.robust_scaled_x_train_OH , self.y_train_OH)
        print('Decision tree best score: {:.4f}'.format(grid_cv.best_score_))
        print('Decision tree best parameter: ', grid_cv.best_params_)
        # 최적의 파라미터값을 만든 모델
        best_df_clf = grid_cv.best_estimator_
        pred_dt = best_df_clf.predict(self.robust_scaled_x_test_OH )
        # decision tree 예측결과
        submission = pd.DataFrame(data=pred_dt, columns=["IsBadBuy"])
        print("Decision tree best estimator prediction:")
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_dt)
        print('Decision Tree best estimator accuracy: {0:.4f}'.format(accuracy))
        # feature의 중요도 plt로 나타내기
        import seaborn as sns
        feature_importance_values = best_df_clf.feature_importances_
        # Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
        feature_importances = pd.Series(feature_importance_values, index=self.X_train_OH.columns)
        # 중요도값 순으로 Series를 정렬
        feature_top20 = feature_importances.sort_values(ascending=False)[:20]
        feature_top1 = feature_top20.index[0]
        feature_top2 = feature_top20.index[1]
        print("feature top1: {0}, feature top2: {1}\n".format(feature_top1, feature_top2))
        plt.figure(figsize=[8, 6])
        plt.title('Feature Importances Top 20')
        sns.barplot(x=feature_top20, y=feature_top20.index)
        plt.show()

        # KNN algorithm evaluation
        from sklearn.neighbors import KNeighborsClassifier
        param_grid = {'n_neighbors': [1, 2, 3, 4, 5]}
        estimator = KNeighborsClassifier()
        grid = GridSearchCV(estimator, param_grid=param_grid)

        grid.fit(self.robust_scaled_x_train_OH , self.y_train_OH)
        print("K-nearest neighbors best score: {0:.4f}".format(grid.best_score_))
        print("best hyperparameter:")
        print(grid.best_params_)
        # KNN 최적의 모델
        best_df_knn = grid.best_estimator_
        # 최적 모델 예측결과
        pred_knn = best_df_knn.predict(self.robust_scaled_x_test_OH )
        submission = pd.DataFrame(data=pred_knn, columns=["IsBadBuy"])
        print("best KNN predict:")
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_knn)
        print('k-nearest neighbor accuracy: {0:.4f}'.format(accuracy))
        print("\n")

        # ensemble learning - Random Forest
        from sklearn.ensemble import RandomForestClassifier
        # instantiate model with 1000 decision trees
        rf = RandomForestClassifier()
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [3, 5, 7, 10],
            'min_samples_split': [2, 3, 5, 10]
        }
        rf_grid = GridSearchCV(rf, param_grid=rf_param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
        rf_grid.fit(self.robust_scaled_x_train_OH , self.y_train_OH)
        print('Random forest best score: {:.4f}'.format(rf_grid.best_score_))
        print('Random forest best parameter: ', rf_grid.best_params_)
        # 최적의 파라미터값을 만든 모델
        best_rf_clf = rf_grid.best_estimator_
        pred_rf = best_rf_clf.predict(self.robust_scaled_x_test_OH )
        # Random Forest 예측결과
        print("Random Forest best estimator prediction")
        submission = pd.DataFrame(data=pred_rf, columns=["IsBadBuy"])
        print(submission)

        accuracy = accuracy_score(self.y_test_OH, pred_rf)
        print('Random Forest accuracy: {0:.4f}'.format(accuracy))
        print("\n")


    def calculateAll(self):
        # 각 조합 순서대로 예측값 계산, 각각 스코어 저장, if score > best_result -> best_score = score

        return 0

    def findBest(self):
        best_result = max(self.scores)
        return best_result

    def showAllResult(self):
        return 0
        # print all results
