# Import the libraries
import numpy as np
import pandas as pd
# For ploting
import matplotlib.pyplot as plt
import seaborn as sns
# For scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
# For encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Reading dataset
dataset = pd.read_csv('training.csv')  # Reading downloaded training csv fife from the program directory
initial_dataset = dataset.copy()  # 혹시몰라서

print(dataset.head())  # Viewing top 5 rows of data set
'''
    RefId  IsBadBuy  PurchDate Auction  VehYear  VehicleAge   Make                Model Trim  ... MMRCurrentRetailCleanPrice PRIMEUNIT AUCGUART  BYRNO VNZIP1  VNST VehBCost IsOnlineSale WarrantyCost
0      1         0  12/7/2009   ADESA     2006           3  MAZDA               MAZDA3    i  ...                    12409.0       NaN      NaN  21973  33619    FL   7100.0            0         1113    
1      2         0  12/7/2009   ADESA     2004           5  DODGE  1500 RAM PICKUP 2WD   ST  ...                    12791.0       NaN      NaN  19638  33619    FL   7600.0            0         1053    
2      3         0  12/7/2009   ADESA     2005           4  DODGE           STRATUS V6  SXT  ...                     8702.0       NaN      NaN  19638  33619    FL   4900.0            0         1389    
3      4         0  12/7/2009   ADESA     2004           5  DODGE                 NEON  SXT  ...                     5518.0       NaN      NaN  19638  33619    FL   4100.0            0          630    
4      5         0  12/7/2009   ADESA     2005           4   FORD                FOCUS  ZX3  ...                     7911.0       NaN      NaN  19638  33619    FL   4000.0            0         1020    
[5 rows x 34 columns]
'''

print(dataset.shape)  # Checking the shape of the dataset, (72983, 34)
print(dataset.columns)  # Checking the features of the dataset
'''
Index(['RefId', 'IsBadBuy', 'PurchDate', 'Auction', 'VehYear', 'VehicleAge',
       'Make', 'Model', 'Trim', 'SubModel', 'Color', 'Transmission',
       'WheelTypeID', 'WheelType', 'VehOdo', 'Nationality', 'Size',
       'TopThreeAmericanName', 'MMRAcquisitionAuctionAveragePrice',
       'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
       'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
       'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
       'MMRCurrentRetailCleanPrice', 'PRIMEUNIT', 'AUCGUART', 'BYRNO',
       'VNZIP1', 'VNST', 'VehBCost', 'IsOnlineSale', 'WarrantyCost'],
      dtype='object')
'''

print(dataset.info())  # Detailed view of data set, showing; data colums, non- null count and data types
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 72983 entries, 0 to 72982
Data columns (total 34 columns):
 #   Column                             Non-Null Count  Dtype
---  ------                             --------------  -----
 0   RefId                              72983 non-null  int64
 1   IsBadBuy                           72983 non-null  int64
 2   PurchDate                          72983 non-null  object
 3   Auction                            72983 non-null  object
 4   VehYear                            72983 non-null  int64
 5   VehicleAge                         72983 non-null  int64
 6   Make                               72983 non-null  object
 7   Model                              72983 non-null  object
 8   Trim                               70623 non-null  object
 9   SubModel                           72975 non-null  object
 10  Color                              72975 non-null  object
 11  Transmission                       72974 non-null  object
 12  WheelTypeID                        69814 non-null  float64
 13  WheelType                          69809 non-null  object
 14  VehOdo                             72983 non-null  int64
 15  Nationality                        72978 non-null  object
 16  Size                               72978 non-null  object
 17  TopThreeAmericanName               72978 non-null  object
 18  MMRAcquisitionAuctionAveragePrice  72965 non-null  float64
 19  MMRAcquisitionAuctionCleanPrice    72965 non-null  float64
 20  MMRAcquisitionRetailAveragePrice   72965 non-null  float64
 21  MMRAcquisitonRetailCleanPrice      72965 non-null  float64
 22  MMRCurrentAuctionAveragePrice      72668 non-null  float64
 23  MMRCurrentAuctionCleanPrice        72668 non-null  float64
 24  MMRCurrentRetailAveragePrice       72668 non-null  float64
 25  MMRCurrentRetailCleanPrice         72668 non-null  float64
 26  PRIMEUNIT                          3419 non-null   object
 27  AUCGUART                           3419 non-null   object
 28  BYRNO                              72983 non-null  int64
 29  VNZIP1                             72983 non-null  int64
 30  VNST                               72983 non-null  object
 31  VehBCost                           72983 non-null  float64
 32  IsOnlineSale                       72983 non-null  int64
 33  WarrantyCost                       72983 non-null  int64
dtypes: float64(10), int64(9), object(15)
memory usage: 18.9+ MB
None
'''

print(dataset.isnull().sum())  # Checking missing values for data
'''
RefId                                    0
IsBadBuy                                 0
PurchDate                                0
Auction                                  0
VehYear                                  0
VehicleAge                               0
Make                                     0
Model                                    0
Trim                                  2360
SubModel                                 8
Color                                    8
Transmission                             9
WheelTypeID                           3169
WheelType                             3174
VehOdo                                   0
Nationality                              5
Size                                     5
TopThreeAmericanName                     5
MMRAcquisitionAuctionAveragePrice       18
MMRAcquisitionAuctionCleanPrice         18
MMRAcquisitionRetailAveragePrice        18
MMRAcquisitonRetailCleanPrice           18
MMRCurrentAuctionAveragePrice          315
MMRCurrentAuctionCleanPrice            315
MMRCurrentRetailAveragePrice           315
MMRCurrentRetailCleanPrice             315
PRIMEUNIT                            69564
AUCGUART                             69564
BYRNO                                    0
VNZIP1                                   0
VNST                                     0
VehBCost                                 0
IsOnlineSale                             0
WarrantyCost                             0
dtype: int64
'''

print(dataset.groupby("IsBadBuy").size())  # Check the distribution of values   in the target column
'''
IsBadBuy
0    64007
1     8976
'''


# Create table for missing data analysis
def find_missing_data(data):
    Total = data.isnull().sum().sort_values(ascending=False)
    Percentage = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)

    return pd.concat([Total, Percentage], axis=1, keys=['Total', 'Percent'])


print(find_missing_data(dataset))
'''
                                   Total   Percent
PRIMEUNIT                          69564  0.953153
AUCGUART                           69564  0.953153
WheelType                           3174  0.043490
WheelTypeID                         3169  0.043421
Trim                                2360  0.032336
MMRCurrentAuctionAveragePrice        315  0.004316
MMRCurrentRetailCleanPrice           315  0.004316
MMRCurrentRetailAveragePrice         315  0.004316
MMRCurrentAuctionCleanPrice          315  0.004316
MMRAcquisitionAuctionAveragePrice     18  0.000247
MMRAcquisitionAuctionCleanPrice       18  0.000247
MMRAcquisitionRetailAveragePrice      18  0.000247
MMRAcquisitonRetailCleanPrice         18  0.000247
Transmission                           9  0.000123
SubModel                               8  0.000110
Color                                  8  0.000110
Nationality                            5  0.000069
Size                                   5  0.000069
TopThreeAmericanName                   5  0.000069
BYRNO                                  0  0.000000
VNZIP1                                 0  0.000000
VNST                                   0  0.000000
VehBCost                               0  0.000000
IsOnlineSale                           0  0.000000
RefId                                  0  0.000000
IsBadBuy                               0  0.000000
VehOdo                                 0  0.000000
Model                                  0  0.000000
Make                                   0  0.000000
VehicleAge                             0  0.000000
VehYear                                0  0.000000
Auction                                0  0.000000
PurchDate                              0  0.000000
WarrantyCost                           0  0.000000
'''

# Checking missing data of dataset using heapamp
sns.heatmap(dataset.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title("Checking missing data using heapamp")
# plt.show()

# Feature removal
# There is too much missing data for features PRIMEUNIT and AUCGUART (missing/all => 69564/72983 = 0.95315347..)
# Therefore, drop the PRIMEUNIT and AUCGUART feature from the dataset
dataset = dataset.drop(['PRIMEUNIT', 'AUCGUART'], axis=1)
# Drop the WheelType feature because you can use WheelTypeID, which is an integer type feature
dataset = dataset.drop(['WheelType'], axis=1)
# Eliminate features unlikely to be useful in the analysis
dataset = dataset.drop(['RefId', 'PurchDate', 'BYRNO'], axis=1)

# Cleaning Missing Data
print(dataset.isnull().sum())  # Checking missing values for data
'''
IsBadBuy                                0
Auction                                 0
VehYear                                 0
VehicleAge                              0
Make                                    0
Model                                   0
Trim                                 2360
SubModel                                8
Color                                   8
Transmission                            9
WheelTypeID                          3169
VehOdo                                  0
Nationality                             5
Size                                    5
TopThreeAmericanName                    5
MMRAcquisitionAuctionAveragePrice      18
MMRAcquisitionAuctionCleanPrice        18
MMRAcquisitionRetailAveragePrice       18
MMRAcquisitonRetailCleanPrice          18
MMRCurrentAuctionAveragePrice         315
MMRCurrentAuctionCleanPrice           315
MMRCurrentRetailAveragePrice          315
MMRCurrentRetailCleanPrice            315
VNZIP1                                  0
VNST                                    0
VehBCost                                0
IsOnlineSale                            0
WarrantyCost                            0
dtype: int64
'''

# The two columns(Trim, WheelTypeID) with quite a few missing values   are filled with a new value 'columnName_unk'
dataset.Trim.fillna(value="Trim_unk", inplace=True)
dataset.WheelTypeID.fillna(value="WheelType_unk", inplace=True)

# For columns with less than 10 missing values, drop rows with missing values
dataset.dropna(subset=['Transmission', 'SubModel', 'Color', 'Nationality', 'Size', 'TopThreeAmericanName'],
               inplace=True)
print(dataset.isnull().sum())  # Checking missing values for data
'''
IsBadBuy                               0
Auction                                0
VehYear                                0
VehicleAge                             0
Make                                   0
Model                                  0
Trim                                   0
SubModel                               0
Color                                  0
Transmission                           0
WheelTypeID                            0
VehOdo                                 0
Nationality                            0
Size                                   0
TopThreeAmericanName                   0
MMRAcquisitionAuctionAveragePrice     18
MMRAcquisitionAuctionCleanPrice       18
MMRAcquisitionRetailAveragePrice      18
MMRAcquisitonRetailCleanPrice         18
MMRCurrentAuctionAveragePrice        311
MMRCurrentAuctionCleanPrice          311
MMRCurrentRetailAveragePrice         311
MMRCurrentRetailCleanPrice           311
VNZIP1                                 0
VNST                                   0
VehBCost                               0
IsOnlineSale                           0
WarrantyCost                           0
dtype: int64
'''

# Before encoding, unify case
print(dataset["Transmission"].value_counts())
'''
AUTO      70393
MANUAL     2575
Manual        1
Name: Transmission, dtype: int64
'''

print(dataset[dataset.Transmission == "Manual"])  # 1rows
'''
Name: Transmission, dtype: int64
       IsBadBuy Auction  VehYear  VehicleAge     Make    Model Trim  ... MMRCurrentRetailAveragePrice MMRCurrentRetailCleanPrice VNZIP1 VNST  VehBCost IsOnlineSale WarrantyCost
33096         0   OTHER     2006           3  HYUNDAI  ELANTRA  GLS  ...                       8331.0                     9199.0  32750   FL    3800.0            0          569
'''

dataset.Transmission.replace("Manual", "MANUAL", inplace=True)  # Replece Manual into MANUAL
print(dataset["Transmission"].value_counts())
'''
AUTO      70393
MANUAL     2576
Name: Transmission, dtype: int64
'''

# Checking outliers using boxplot
# Collect and check columns with similar ranges among integer type columns
sns.boxplot(data=dataset.loc[:, 'MMRAcquisitionAuctionAveragePrice':'MMRAcquisitonRetailCleanPrice'])
plt.title("Checking outliers using boxplot (1)")
# plt.show()

sns.boxplot(data=dataset.loc[:, 'MMRCurrentAuctionAveragePrice':'MMRCurrentRetailCleanPrice'])
plt.title("Checking outliers using boxplot (2)")
# plt.show()

sns.boxplot(data=dataset.loc[:, ['VehOdo', 'VNZIP1', 'VehBCost']])
plt.title("Checking outliers using boxplot (3)")
# plt.show()

# As a result of checking outliers, it is judged that the values   are too distributed to help predict
# So drop these features
dataset.drop(
    ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
     'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
     'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice'], axis=1, inplace=True)
print(dataset.isnull().sum())  # Checking missing values for data
'''
IsBadBuy                0
Auction                 0
VehYear                 0
VehicleAge              0
Make                    0
Model                   0
Trim                    0
SubModel                0
Color                   0
Transmission            0
WheelTypeID             0
VehOdo                  0
Nationality             0
Size                    0
TopThreeAmericanName    0
VNZIP1                  0
VNST                    0
VehBCost                0
IsOnlineSale            0
WarrantyCost            0
dtype: int64
'''

import opensource

newDatset = dataset.copy()

easyCombination = opensource.EasyCombination(newDatset)
easyCombination.encodeAndSplit('IsBadBuy')
easyCombination.scale()


# Decision tree evaluation
easyCombination.estimate(1)
easyCombination.findBestScore()
easyCombination.findWorstScore()

'''
Selected Decision Tree
Score using Label Encoding and Standard Scaling: 0.8190123795166964
Score using Label Encoding and MinMax Scaling: 0.8189666986432781
Score using Label Encoding and Maxabs Scaling: 0.8190123795166964
Score using Label Encoding and Robust Scaling: 0.8189666986432781
Score using One-Hot Encoding and Standard Scaling: 0.8190123795166964
Score using One-Hot Encoding and MinMax Scaling: 0.8189666986432781
Score using One-Hot Encoding and Maxabs Scaling: 0.8190123795166964
Score using One-Hot Encoding and Robust Scaling: 0.8189666986432781 


Best result score is: 0.8190123795166964
Using Label Encoding and Standard Scaling
Using Label Encoding and Maxabs Scaling
Using One-Hot Encoding and Standard Scaling
Using One-Hot Encoding and Maxabs Scaling

Worst result score is: 0.8189666986432781
Using Label Encoding and MinMax Scaling
Using Label Encoding and Robust Scaling
Using One-Hot Encoding and MinMax Scaling
Using One-Hot Encoding and Robust Scaling
'''

# We use Label Encoder and Standard Scaler
# Convert Categorical Value to Integer Value through Encoding
# Label encoding usging a label encoder
dataset_dt = dataset.copy()
label_encoder = LabelEncoder()
for i in dataset_dt.columns[dataset_dt.dtypes==object]:
    dataset_dt[i]=label_encoder.fit_transform(list(dataset_dt[i]))

print(dataset_dt.head())
'''
   IsBadBuy  Auction  VehYear  VehicleAge  Make  Model  Trim  SubModel  Color  Transmission  WheelTypeID  VehOdo  Nationality  Size  TopThreeAmericanName  VNZIP1  VNST  VehBCost  IsOnlineSale  WarrantyCost
0         0        0     2006           3    17    586   133       221     12             0            1   89046            2     5                     3   33619     5    7100.0             0          1113 
1         0        0     2004           5     5      0    93       764     14             0            1   93593            0     4                     0   33619     5    7600.0             0          1053 
2         0        0     2005           4     5    882    98       292      7             0            2   73807            0     5                     0   33619     5    4900.0             0          1389 
3         0        0     2004           5     5    662    98       152     13             0            1   65617            0     0                     0   33619     5    4100.0             0           630 
4         0        0     2005           4     6    368   127        52     13             1            2   69367            0     0                     1   33619     5    4000.0             0          1020 
'''

print(dataset_dt.info())
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 72969 entries, 0 to 72982
Data columns (total 20 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   IsBadBuy              72969 non-null  int64
 1   Auction               72969 non-null  int64
 2   VehYear               72969 non-null  int64
 3   VehicleAge            72969 non-null  int64
 4   Make                  72969 non-null  int64
 5   Model                 72969 non-null  int64
 6   Trim                  72969 non-null  int64
 7   SubModel              72969 non-null  int64
 8   Color                 72969 non-null  int64
 9   Transmission          72969 non-null  int64
 10  WheelTypeID           72969 non-null  int64
 11  VehOdo                72969 non-null  int64
 12  Nationality           72969 non-null  int64
 13  Size                  72969 non-null  int64
 14  TopThreeAmericanName  72969 non-null  int64
 15  VNZIP1                72969 non-null  int64
 16  VNST                  72969 non-null  int64
 17  VehBCost              72969 non-null  float64
 18  IsOnlineSale          72969 non-null  int64
 19  WarrantyCost          72969 non-null  int64
dtypes: float64(1), int64(19)
memory usage: 11.7 MB
None
'''

# Splitting the dataset into Training and Test Set
X = dataset_dt.drop(['IsBadBuy'], axis=1)
y = dataset_dt['IsBadBuy']  # IsBadBuy is the target feature

# The test set is allocated 30% of the total data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)  # (51078, 19) (51078,)
print(X_test.shape, y_test.shape)  # (21891, 19) (21891,)

# Feature scaling
# If the data is not normalized, the pattern of change may not be detected
standard_scaler = StandardScaler()  # Using standard scaler
standard_scaler.fit(X_train)
X_train_scaled = standard_scaler.transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)  # Converts back to dataframe after scaling
X_test_scaled = standard_scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)  # Converts back to dataframe after scaling

print(X_train_scaled)

'''
        Auction   VehYear  VehicleAge      Make     Model      Trim  SubModel     Color  ...  Nationality      Size  TopThreeAmericanName    VNZIP1      VNST  VehBCost  IsOnlineSale  WarrantyCost
0     -0.063839  0.381494   -0.106359  1.925582 -0.375270 -1.398669 -0.234648 -1.157847  ...    -0.430816  0.069751              0.590831 -0.420274 -0.525757 -0.364077     -0.160324     -0.397782
1      1.450677  1.537006   -1.274279 -0.493618 -1.452535  0.627837 -0.801535 -1.352340  ...    -0.430816  0.069751             -1.238221  0.613301  0.832810  0.610754     -0.160324     -0.595525
2     -0.063839 -0.774017    0.477602 -0.748271  1.516085  0.222536  0.453715 -0.185381  ...    -0.430816  0.424432              0.590831  1.030658 -1.165083  0.329389     -0.160324     -0.201715
3     -1.578355  0.381494   -0.690319 -0.620944  0.603743 -1.398669  0.038673  0.981578  ...    -0.430816 -1.348970             -1.238221 -1.067722 -0.845420  0.809700     -0.160324     -0.503356
4     -0.063839  0.959250   -0.690319 -0.748271  1.617846 -0.101705  2.194869  0.981578  ...    -0.430816  2.197833              0.590831  0.089166 -0.605673  0.599386      6.237362      1.366820
...         ...       ...         ...       ...       ...       ...       ...       ...  ...          ...       ...                   ...       ...       ...       ...           ...           ...
51073 -0.063839 -0.196261    0.477602 -0.620944  0.705504 -0.669126 -1.565821  0.981578  ...    -0.430816  0.069751             -1.238221  0.655439  1.232388 -0.315762     -0.160324     -0.178254
51074 -0.063839  0.959250   -1.274279 -0.748271 -0.140167 -0.101705 -0.396616  1.176071  ...    -0.430816 -0.994289              0.590831  0.231576  0.033653  0.130444     -0.160324      1.469043
51075 -0.063839  1.537006   -1.274279 -0.748271 -0.105077 -0.101705 -0.386493 -1.157847  ...    -0.430816 -0.994289              0.590831 -1.093747  1.072557  1.014329     -0.160324      1.469043
51076 -0.063839 -0.196261   -0.106359 -0.493618 -1.813963  0.843998  2.296099  0.981578  ...    -0.430816 -0.284929             -1.238221 -1.351356  1.392219  0.588018     -0.160324     -0.372645
51077 -1.578355  0.381494   -0.690319 -0.366292 -0.529666  0.627837 -0.234648 -0.379874  ...    -0.430816 -0.994289             -0.323695  0.869035 -1.005251 -0.162290     -0.160324      0.386486

[51078 rows x 19 columns]
'''

print(X_test_scaled)
'''
        Auction   VehYear  VehicleAge      Make     Model      Trim  SubModel     Color  ...  Nationality      Size  TopThreeAmericanName    VNZIP1      VNST  VehBCost  IsOnlineSale  WarrantyCost
0     -1.578355  0.381494   -0.690319 -0.748271 -0.129640 -0.101705 -0.396616 -1.157847  ...    -0.430816 -0.994289              0.590831  0.869035 -1.005251  0.269705     -0.160324      0.376431
1     -1.578355  0.381494   -0.690319  2.434887  0.859901  1.303339 -0.801535  0.203605  ...     1.862351 -1.703650              1.505356 -1.141931  0.193484 -0.682389     -0.160324     -1.363034
2     -1.578355 -0.196261   -0.106359 -0.748271  1.516085 -0.101705  0.448654 -1.157847  ...    -0.430816  0.424432              0.590831  1.037776 -1.165083  1.062644     -0.160324     -0.595525
3     -0.063839 -0.196261   -0.106359 -0.493618 -0.940220  0.925058  2.240423  1.176071  ...    -0.430816  1.133792             -1.238221  0.838188 -1.005251 -0.210605     -0.160324     -0.595525
4      1.450677  0.381494   -0.106359 -0.493618  1.266946  1.060158 -0.801535 -0.574367  ...    -0.430816  0.069751             -1.238221 -1.087432  1.072557 -0.412392     -0.160324     -0.054246
...         ...       ...         ...       ...       ...       ...       ...       ...  ...          ...       ...                   ...       ...       ...       ...           ...           ...
21886  1.450677 -0.774017    0.477602  2.052908 -0.055950 -1.587809 -0.426985 -0.185381  ...    -0.430816  0.069751              0.590831  1.437604 -1.085167 -1.327540     -0.160324     -0.917275
21887 -0.063839  0.381494   -0.106359 -0.748271  0.137045 -0.101705 -0.396616  1.176071  ...    -0.430816  0.069751              0.590831  0.655439  1.232388  0.082129     -0.160324     -0.397782
21888 -0.063839  0.381494   -0.106359 -0.748271 -1.129707 -0.101705 -0.396616  1.176071  ...    -0.430816 -1.703650              0.590831  1.327917 -1.085167 -0.622706     -0.160324     -0.917275
21889 -0.063839  0.381494   -0.106359 -0.748271 -1.129707 -0.074685 -1.398792 -1.352340  ...    -0.430816 -1.703650              0.590831 -0.716002  0.113568 -0.165132     -0.160324     -0.917275
21890 -1.578355 -0.196261    0.477602 -0.620944  0.705504  1.276319 -0.801535 -1.157847  ...    -0.430816  0.069751             -1.238221 -0.778423  1.152472 -0.548812     -0.160324     -0.317344

[21891 rows x 19 columns]
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
params = {
    'max_depth' : [6, 8, 10, 12, 16, 20, 24],
    'min_samples_split' : [16, 24]
}
DecisionTree = DecisionTreeClassifier(random_state=156)
grid_cv = GridSearchCV(DecisionTree,param_grid=params,scoring ='accuracy',cv=5,verbose=1)
grid_cv.fit(X_train_scaled, y_train)
print('Decision tree best score: {:.4f}'.format(grid_cv.best_score_))
print('Decision tree best parameter: ', grid_cv.best_params_)
# 최적의 파라미터값을 만든 모델
best_df_clf = grid_cv.best_estimator_
pred_dt = best_df_clf.predict(X_test_scaled)
#decision tree 예측결과
submission = pd.DataFrame(data=pred_dt,columns=["IsBadBuy"])
print("Decision tree best estimator prediction:")
print(submission)
'''
Decision tree best estimator prediction:
       IsBadBuy
0             0
1             0
2             0
3             0
4             0
...         ...
21886         0
21887         0
21888         0
21889         0
21890         0

[21891 rows x 1 columns]
'''
accuracy = accuracy_score(y_test, pred_dt) 
print('Decision Tree best estimator accuracy: {0:.4f}'.format(accuracy)) #Decision Tree best estimator accuracy: 0.9030

#feature의 중요도 plt로 나타내기
import seaborn as sns
feature_importance_values = best_df_clf.feature_importances_
# Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
feature_importances = pd.Series(feature_importance_values, index=X_train.columns)
# 중요도값 순으로 Series를 정렬
feature_top20 = feature_importances.sort_values(ascending=False)[:20]
feature_top1 = feature_top20.index[0]
feature_top2 = feature_top20.index[1]
print("feature top1: {0}, feature top2: {1}\n".format(feature_top1,feature_top2)) # feature top1: WheelTypeID, feature top2: VehicleAge
plt.figure(figsize=[8, 6])
plt.title('Feature Importances Top 20')
sns.barplot(x=feature_top20, y=feature_top20.index)
plt.show()


# KNN algorithm evaluation
easyCombination.estimate(2)
easyCombination.findBestScore()
easyCombination.findWorstScore()
'''
Selected K-nearest neighbors
Score using Label Encoding and Standard Scaling: 0.8896350098213878
Score using Label Encoding and MinMax Scaling: 0.8881275409985839
Score using Label Encoding and Maxabs Scaling: 0.8898634141884793
Score using Label Encoding and Robust Scaling: 0.8913252021378649
Score using One-Hot and Standard Scaling: 0.8896350098213878
Score using One-Hot Encoding and MinMax Scaling: 0.8881275409985839
Score using One-Hot Encoding and Maxabs Scaling: 0.8898634141884793
Score using One-Hot Encoding and Robust Scaling: 0.8913252021378649 


Best result score is: 0.8913252021378649
Using Label Encoding and Robust Scaling
Using One-Hot Encoding and Robust Scaling

Worst result score is: 0.8881275409985839
Using Label Encoding and MinMax Scaling
Using One-Hot Encoding and MinMax Scaling
'''

# We use One-Hot Encoder and Robust scaler
# Convert Categorical Value to Integer Value through Encoding
# One-Hot encoding using 'get_dummies' function
dataset_knn = dataset.copy()
print(dataset_knn["Trim"].value_counts())
'''
Bas    13946
LS     10174
SE      9346
SXT     3822
LT      3540
       ...
Z24        1
Har        1
Xsp        1
JLX        1
Ult        1
Name: Trim, Length: 135, dtype: int64
'''

print(dataset_knn["Model"].value_counts())
'''
PT CRUISER              2329
IMPALA                  1990
TAURUS                  1425
CALIBER                 1375
CARAVAN GRAND FWD V6    1289
                        ...
RELAY 2WD V6 3.9L V6       1
350Z MFI V6 3.5L DOH       1
FJ CRUISER 4WD V6          1
GX470 4WD                  1
RAINIER AWD V8             1
Name: Model, Length: 1062, dtype: int64
'''

print(dataset_knn["SubModel"].value_counts())
'''
4D SEDAN                15235
4D SEDAN LS              4718
4D SEDAN SE              3859
4D WAGON                 2230
MINIVAN 3.3L             1258
                        ...
4D SPORT UTILITY JLX        1
REG CAB 4.7L ST             1
MINIVAN LX                  1
4D SUV 5.9L SLT PLUS        1
4D EXT CAB 3.0L XL          1
Name: SubModel, Length: 862, dtype: int64
'''

# There are too many categories for one-hot encoding. Therefore, in this situation, the Trim, Model, and SubModel features are dropped
dataset_knn.drop(['Trim', 'Model', 'SubModel'], axis=1, inplace=True)

for i in dataset_knn.columns[dataset_knn.dtypes == object]:
    dataset_knn = pd.get_dummies(data=dataset_knn, columns=[i], prefix=i)

print(dataset_knn.head())
'''
   IsBadBuy  VehYear  VehicleAge  VehOdo  VNZIP1  VehBCost  IsOnlineSale  WarrantyCost  Auction_ADESA  ...  VNST_OR  VNST_PA  VNST_SC  VNST_TN  VNST_TX  VNST_UT  VNST_VA  VNST_WA  VNST_WV
0         0     2006           3   89046   33619    7100.0             0          1113              1  ...        0        0        0        0        0        0        0        0        0
1         0     2004           5   93593   33619    7600.0             0          1053              1  ...        0        0        0        0        0        0        0        0        0
2         0     2005           4   73807   33619    4900.0             0          1389              1  ...        0        0        0        0        0        0        0        0        0
3         0     2004           5   65617   33619    4100.0             0           630              1  ...        0        0        0        0        0        0        0        0        0
4         0     2005           4   69367   33619    4000.0             0          1020              1  ...        0        0        0        0        0        0        0        0        0

[5 rows x 124 columns]
'''

print(dataset_knn.info())
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 72969 entries, 0 to 72982
Columns: 124 entries, IsBadBuy to VNST_WV
dtypes: float64(1), int64(7), uint8(116)
memory usage: 13.1 MB
None
'''

# Splitting the dataset into Training and Test Set
X = dataset_knn.drop(['IsBadBuy'], axis=1)
y = dataset_knn['IsBadBuy']  # IsBadBuy is the target feature

# The test set is allocated 30% of the total data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)  # (51078, 123) (51078,)
print(X_test.shape, y_test.shape)  # (21891, 123) (21891,)

# Feature scaling
# If the data is not normalized, the pattern of change may not be detected
robust_scaler = RobustScaler()  # Using standard scaler
robust_scaler.fit(X_train)
X_train_scaled = robust_scaler.transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)  # Converts back to dataframe after scaling
X_test_scaled = robust_scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)  # Converts back to dataframe after scaling

print(X_train_scaled)

'''
        VehYear  VehicleAge    VehOdo    VNZIP1  VehBCost  IsOnlineSale  WarrantyCost  Auction_ADESA  Auction_MANHEIM  ...  VNST_OR  VNST_PA  VNST_SC  VNST_TN  VNST_TX  VNST_UT  VNST_VA  VNST_WA  VNST_WV
0      0.333333         0.0  0.304330 -0.563823 -0.250505           0.0     -0.148855            0.0              0.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
1      1.000000        -1.0 -1.084657  0.000000  0.442424           0.0     -0.298982            0.0             -1.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
2     -0.333333         0.5  0.234229  0.227671  0.242424           0.0      0.000000            0.0              0.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
3      0.333333        -0.5 -1.524949 -0.917011  0.583838           0.0     -0.229008            1.0             -1.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
4      0.666667        -0.5  0.162043 -0.285920  0.434343           1.0      1.190840            0.0              0.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
...         ...         ...       ...       ...       ...           ...           ...            ...              ...  ...      ...      ...      ...      ...      ...      ...      ...      ...      ...   
51073  0.000000         0.5 -0.665843  0.022986 -0.216162           0.0      0.017812            0.0              0.0  ...      0.0      0.0      0.0      0.0      1.0      0.0      0.0      0.0      0.0   
51074  0.666667        -1.0  0.753978 -0.208234  0.101010           0.0      1.268448            0.0              0.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
51075  1.000000        -1.0  0.731726 -0.931208  0.729293           0.0      1.268448            0.0              0.0  ...      0.0      0.0      1.0      0.0      0.0      0.0      0.0      0.0      0.0   
51076  0.000000         0.0  0.820541 -1.071736  0.426263           0.0     -0.129771            0.0              0.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      1.0      0.0      0.0   
51077  0.333333        -0.5 -0.061351  0.139505 -0.107071           0.0      0.446565            1.0             -1.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   

[51078 rows x 123 columns]
'''

print(X_test_scaled)
'''
        VehYear  VehicleAge    VehOdo    VNZIP1  VehBCost  IsOnlineSale  WarrantyCost  Auction_ADESA  Auction_MANHEIM  ...  VNST_OR  VNST_PA  VNST_SC  VNST_TN  VNST_TX  VNST_UT  VNST_VA  VNST_WA  VNST_WV
0      0.333333        -0.5 -1.084560  0.139505  0.200000           0.0      0.438931            1.0             -1.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
1      0.333333        -0.5 -1.628744 -0.957493 -0.476768           0.0     -0.881679            1.0             -1.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
2      0.000000         0.0 -0.722224  0.231555  0.763636           0.0     -0.298982            1.0             -1.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
3      0.000000         0.0 -0.838527  0.122677 -0.141414           0.0     -0.298982            0.0              0.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
4      0.333333         0.0  0.582603 -0.927763 -0.284848           0.0      0.111959            0.0             -1.0  ...      0.0      0.0      1.0      0.0      0.0      0.0      0.0      0.0      0.0   
...         ...         ...       ...       ...       ...           ...           ...            ...              ...  ...      ...      ...      ...      ...      ...      ...      ...      ...      ...   
21886 -0.333333         0.5 -0.189822  0.449664 -0.935354           0.0     -0.543257            0.0             -1.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
21887  0.333333         0.0  0.218231  0.022986  0.066667           0.0     -0.148855            0.0              0.0  ...      0.0      0.0      0.0      0.0      1.0      0.0      0.0      0.0      0.0   
21888  0.333333         0.0 -0.225260  0.389828 -0.434343           0.0     -0.543257            0.0              0.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
21889  0.333333         0.0 -0.555115 -0.725145 -0.109091           0.0     -0.543257            0.0              0.0  ...      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   
21890  0.000000         0.5 -0.862573 -0.759197 -0.381818           0.0     -0.087786            1.0             -1.0  ...      0.0      0.0      0.0      1.0      0.0      0.0      0.0      0.0      0.0   

[21891 rows x 123 columns]
'''

from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_neighbors':[1,2,3,4,5]}
estimator = KNeighborsClassifier()
grid = GridSearchCV(estimator, param_grid=param_grid)

grid.fit(X_train_scaled, y_train)
print("K-nearest neighbors best score: {0:.4f}".format(grid.best_score_)) # K-nearest neighbors best score: 0.8804
print("best hyperparameter:") # {'n_neighbors': 4}
print(grid.best_params_)
# KNN 최적의 모델
best_df_knn = grid.best_estimator_
# 최적 모델 예측결과
pred_knn = best_df_knn.predict(X_test_scaled)
submission = pd.DataFrame(data=pred_knn,columns=["IsBadBuy"])
print("best KNN predict:")
print(submission)
'''
best KNN predict:
       IsBadBuy
0             0
1             0
2             0
3             0
4             0
...         ...
21886         0
21887         0
21888         0
21889         0
21890         0

[21891 rows x 1 columns]
'''
accuracy = accuracy_score(y_test, pred_knn)
print('k-nearest neighbor accuracy: {0:.4f}'.format(accuracy)) # k-nearest neighbor accuracy: 0.8846
print("\n")



# ensemble learning - Random Forest
easyCombination.estimate(3)
easyCombination.findBestScore()
easyCombination.findWorstScore()

'''
Selected Random Forest
Score using Label Encoding and Standard Scaling: 0.9009181855557078
Score using Label Encoding and MinMax Scaling: 0.9002786533278516
Score using Label Encoding and Robust Scaling: 0.8998675254670869
Score using One-Hot and Standard Scaling: 0.9005527385683614
Score using One-Hot Encoding and MinMax Scaling: 0.9008268238088712
Score using One-Hot Encoding and Maxabs Scaling: 0.9007354620620346
Score using One-Hot Encoding and Robust Scaling: 0.9005984194417798


Best result score is: 0.9009181855557078
Using Label Encoding and Standard Scaling

Worst result score is: 0.8998675254670869
Using Label Encoding and Robust Scaling
'''

# We use Label Encoder and Robust Scaler
# Convert Categorical Value to Integer Value through Encoding
# Label encoding usging a label encoder
dataset_rs = dataset.copy()
label_encoder = LabelEncoder()
for i in dataset_rs.columns[dataset_rs.dtypes==object]:
    dataset_rs[i]=label_encoder.fit_transform(list(dataset_rs[i]))

print(dataset_rs.head())
'''
   IsBadBuy  Auction  VehYear  VehicleAge  Make  Model  Trim  SubModel  Color  Transmission  WheelTypeID  VehOdo  Nationality  Size  TopThreeAmericanName  VNZIP1  VNST  VehBCost  IsOnlineSale  WarrantyCost
0         0        0     2006           3    17    586   133       221     12             0            1   89046            2     5                     3   33619     5    7100.0             0          1113 
1         0        0     2004           5     5      0    93       764     14             0            1   93593            0     4                     0   33619     5    7600.0             0          1053 
2         0        0     2005           4     5    882    98       292      7             0            2   73807            0     5                     0   33619     5    4900.0             0          1389 
3         0        0     2004           5     5    662    98       152     13             0            1   65617            0     0                     0   33619     5    4100.0             0           630 
4         0        0     2005           4     6    368   127        52     13             1            2   69367            0     0                     1   33619     5    4000.0             0          1020 
'''

print(dataset_rs.info())
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 72969 entries, 0 to 72982
Data columns (total 20 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   IsBadBuy              72969 non-null  int64
 1   Auction               72969 non-null  int64
 2   VehYear               72969 non-null  int64
 3   VehicleAge            72969 non-null  int64
 4   Make                  72969 non-null  int64
 5   Model                 72969 non-null  int64
 6   Trim                  72969 non-null  int64
 7   SubModel              72969 non-null  int64
 8   Color                 72969 non-null  int64
 9   Transmission          72969 non-null  int64
 10  WheelTypeID           72969 non-null  int64
 11  VehOdo                72969 non-null  int64
 12  Nationality           72969 non-null  int64
 13  Size                  72969 non-null  int64
 14  TopThreeAmericanName  72969 non-null  int64
 15  VNZIP1                72969 non-null  int64
 16  VNST                  72969 non-null  int64
 17  VehBCost              72969 non-null  float64
 18  IsOnlineSale          72969 non-null  int64
 19  WarrantyCost          72969 non-null  int64
dtypes: float64(1), int64(19)
memory usage: 11.7 MB
None
'''

# Splitting the dataset into Training and Test Set
X = dataset_rs.drop(['IsBadBuy'], axis=1)
y = dataset_rs['IsBadBuy']  # IsBadBuy is the target feature

# The test set is allocated 30% of the total data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)  # (51078, 19) (51078,)
print(X_test.shape, y_test.shape)  # (21891, 19) (21891,)

# Feature scaling
# If the data is not normalized, the pattern of change may not be detected
robust_scaler = RobustScaler()  # Using standard scaler
robust_scaler.fit(X_train)
X_train_scaled = robust_scaler.transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)  # Converts back to dataframe after scaling
X_test_scaled = robust_scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)  # Converts back to dataframe after scaling

print(X_train_scaled)

'''
       Auction   VehYear  VehicleAge      Make     Model      Trim  SubModel     Color  ...  Nationality      Size  TopThreeAmericanName    VNZIP1      VNST  VehBCost  IsOnlineSale  WarrantyCost
0          0.0  0.333333         0.0  2.111111 -0.169978 -0.790323  0.088710 -0.454545  ...          0.0  0.000000                   0.0 -0.563823 -0.296296 -0.250505           0.0     -0.148855
1          1.0  1.000000        -1.0  0.000000 -0.847682  0.419355 -0.362903 -0.545455  ...          0.0  0.000000                  -1.0  0.000000  0.333333  0.442424           0.0     -0.298982
2          0.0 -0.333333         0.5 -0.222222  1.019868  0.177419  0.637097  0.000000  ...          0.0  0.333333                   0.0  0.227671 -0.592593  0.242424           0.0      0.000000
3         -1.0  0.333333        -0.5 -0.111111  0.445916 -0.790323  0.306452  0.545455  ...          0.0 -1.333333                  -1.0 -0.917011 -0.444444  0.583838           0.0     -0.229008
4          0.0  0.666667        -0.5 -0.222222  1.083885 -0.016129  2.024194  0.545455  ...          0.0  2.000000                   0.0 -0.285920 -0.333333  0.434343           1.0      1.190840
...        ...       ...         ...       ...       ...       ...       ...       ...  ...          ...       ...                   ...       ...       ...       ...           ...           ...
51073      0.0  0.000000         0.5 -0.111111  0.509934 -0.354839 -0.971774  0.545455  ...          0.0  0.000000                  -1.0  0.022986  0.518519 -0.216162           0.0      0.017812
51074      0.0  0.666667        -1.0 -0.222222 -0.022075 -0.016129 -0.040323  0.636364  ...          0.0 -1.000000                   0.0 -0.208234 -0.037037  0.101010           0.0      1.268448
51075      0.0  1.000000        -1.0 -0.222222  0.000000 -0.016129 -0.032258 -0.454545  ...          0.0 -1.000000                   0.0 -0.931208  0.444444  0.729293           0.0      1.268448
51076      0.0  0.000000         0.0  0.000000 -1.075055  0.548387  2.104839  0.545455  ...          0.0 -0.333333                  -1.0 -1.071736  0.592593  0.426263           0.0     -0.129771
51077     -1.0  0.333333        -0.5  0.111111 -0.267108  0.419355  0.088710 -0.090909  ...          0.0 -1.000000                  -0.5  0.139505 -0.518519 -0.107071           0.0      0.446565

[51078 rows x 19 columns]
'''

print(X_test_scaled)
'''
       Auction   VehYear  VehicleAge      Make     Model      Trim  SubModel     Color  ...  Nationality      Size  TopThreeAmericanName    VNZIP1      VNST  VehBCost  IsOnlineSale  WarrantyCost
0         -1.0  0.333333        -0.5 -0.222222 -0.015453 -0.016129 -0.040323 -0.454545  ...          0.0 -1.000000                   0.0  0.139505 -0.518519  0.200000           0.0      0.438931
1         -1.0  0.333333        -0.5  2.555556  0.607064  0.822581 -0.362903  0.181818  ...          2.0 -1.666667                   0.5 -0.957493  0.037037 -0.476768           0.0     -0.881679
2         -1.0  0.000000         0.0 -0.222222  1.019868 -0.016129  0.633065 -0.454545  ...          0.0  0.333333                   0.0  0.231555 -0.592593  0.763636           0.0     -0.298982
3          0.0  0.000000         0.0  0.000000 -0.525386  0.596774  2.060484  0.636364  ...          0.0  1.000000                  -1.0  0.122677 -0.518519 -0.141414           0.0     -0.298982
4          1.0  0.333333         0.0  0.000000  0.863135  0.677419 -0.362903 -0.181818  ...          0.0  0.000000                  -1.0 -0.927763  0.444444 -0.284848           0.0      0.111959
...        ...       ...         ...       ...       ...       ...       ...       ...  ...          ...       ...                   ...       ...       ...       ...           ...           ...
21886      1.0 -0.333333         0.5  2.222222  0.030905 -0.903226 -0.064516  0.000000  ...          0.0  0.000000                   0.0  0.449664 -0.555556 -0.935354           0.0     -0.543257
21887      0.0  0.333333         0.0 -0.222222  0.152318 -0.016129 -0.040323  0.636364  ...          0.0  0.000000                   0.0  0.022986  0.518519  0.066667           0.0     -0.148855
21888      0.0  0.333333         0.0 -0.222222 -0.644592 -0.016129 -0.040323  0.636364  ...          0.0 -1.666667                   0.0  0.389828 -0.555556 -0.434343           0.0     -0.543257
21889      0.0  0.333333         0.0 -0.222222 -0.644592  0.000000 -0.838710 -0.545455  ...          0.0 -1.666667                   0.0 -0.725145  0.000000 -0.109091           0.0     -0.543257
21890     -1.0  0.000000         0.5 -0.111111  0.509934  0.806452 -0.362903 -0.454545  ...          0.0  0.000000                  -1.0 -0.759197  0.481481 -0.381818           0.0     -0.087786

[21891 rows x 19 columns]
'''

from sklearn.ensemble import RandomForestClassifier
# instantiate model with 1000 decision trees
rf = RandomForestClassifier()
rf_param_grid = {
    'n_estimators': [100,200],
    'max_depth': [6,8,10,12],
    'min_samples_leaf':[3,5,7,10],
    'min_samples_split':[2,3,5,10]
}
rf_grid = GridSearchCV(rf,param_grid=rf_param_grid,scoring='accuracy',n_jobs=-1,verbose = 1)
rf_grid.fit(X_train_scaled,y_train)
print('Random forest best score: {:.4f}'.format(rf_grid.best_score_)) # Random forest best score: 0.8983
print('Random forest best parameter: ', rf_grid.best_params_) # Random forest best parameter:  {'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 200}
# 최적의 파라미터값을 만든 모델
best_rf_clf = rf_grid.best_estimator_
pred_rf = best_rf_clf.predict(X_test_scaled)
# Random Forest 예측결과
print("Random Forest best estimator prediction")
submission = pd.DataFrame(data=pred_rf,columns=["IsBadBuy"])
print(submission)
'''
Random Forest best estimator prediction
       IsBadBuy
0             0
1             0
2             0
3             0
4             0
...         ...
21886         0
21887         0
21888         0
21889         0
21890         0

[21891 rows x 1 columns]
'''

accuracy = accuracy_score(y_test, pred_rf)
print('Random Forest accuracy: {0:.4f}'.format(accuracy)) # Random Forest accuracy: 0.9022
print("\n")

"""
 # random forest 출력 결과
Fitting 5 folds for each of 128 candidates, totalling 640 fits
Random forest 최고 평균 정확도 수치: 0.8984
random forest 최적 하이퍼파라미터:  {'max_depth': 12, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 200}
Decision tree 최적 예측 결과
[0 0 0 ... 0 0 0]
Decision Tree 예측 정확도: 0.9022
"""