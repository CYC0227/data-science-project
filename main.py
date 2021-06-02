# Import the libraries
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

print(dataset.groupby("IsBadBuy").size())  # Check the distribution of values ​​in the target column
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
plt.show()

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

# The two columns(Trim, WheelTypeID) with quite a few missing values ​​are filled with a new value 'columnName_unk'
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
plt.show()

sns.boxplot(data=dataset.loc[:, 'MMRCurrentAuctionAveragePrice':'MMRCurrentRetailCleanPrice'])
plt.title("Checking outliers using boxplot (2)")
plt.show()

sns.boxplot(data=dataset.loc[:, ['VehOdo', 'VNZIP1', 'VehBCost']])
plt.title("Checking outliers using boxplot (3)")
plt.show()

# As a result of checking outliers, it is judged that the values ​​are too distributed to help predict
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

# Convert Categorical Value to Integer Value through Encoding
# Label encoding usging a label encoder
# label_encoder = LabelEncoder()
# for i in dataset.columns[dataset.dtypes==object]:
#     print(i)
#     dataset[i]=label_encoder.fit_transform(list(dataset[i]))

# One-Hot encoding using 'get_dummies' function
print(dataset["Trim"].value_counts())
'''
Bas    13946
LS     10174
SE      9346
SXT     3822
LT      3540
       ...
Har        1
JLS        1
Ult        1
Z24        1
Out        1
Name: Trim, Length: 135, dtype: int64
'''

print(dataset["Model"].value_counts())
'''
model PT CRUISER        2329
IMPALA                  1990
TAURUS                  1425
CALIBER                 1375
CARAVAN GRAND FWD V6    1289
                        ...
ECLIPSE EI V6 3.0L S       1
SILHOUETTE 3.4L V 6        1
FIT                        1
PATRIOT 2WD 4C 2.0L        1
ALERO 4C 2.4L I-4 SF       1
Name: Model, Length: 1062, dtype: int64
'''

print(dataset["SubModel"].value_counts())
'''
4D SEDAN                           15235
4D SEDAN LS                         4718
4D SEDAN SE                         3859
4D WAGON                            2230
MINIVAN 3.3L                        1258
                                     ...
PASSENGER 3.4L LS                      1
EXT CAB 8.1L LS                        1
ACCESS CAB 4.0L SR5                    1
EXT CAB 4.0L SE                        1
4D SPORT UTILITY HYBRID LIMITED        1
Name: SubModel, Length: 862, dtype: int64
'''

# There are too many categories for one-hot encoding. Therefore, in this situation, the Trim, Model, and SubModel features are dropped
dataset.drop(['Trim', 'Model', 'SubModel'], axis=1, inplace=True)

for i in dataset.columns[dataset.dtypes == object]:
    print(i)
    dataset = pd.get_dummies(data=dataset, columns=[i], prefix=i)

print(dataset.head())
'''
   IsBadBuy  VehYear  VehicleAge  VehOdo  VNZIP1  VehBCost  IsOnlineSale  WarrantyCost  Auction_ADESA  ...  VNST_OR  VNST_PA  VNST_SC  VNST_TN  VNST_TX  VNST_UT  VNST_VA  VNST_WA  VNST_WV
0         0     2006           3   89046   33619    7100.0             0          1113              1  ...        0        0        0        0        0        0        0        0        0
1         0     2004           5   93593   33619    7600.0             0          1053              1  ...        0        0        0        0        0        0        0        0        0
2         0     2005           4   73807   33619    4900.0             0          1389              1  ...        0        0        0        0        0        0        0        0        0
3         0     2004           5   65617   33619    4100.0             0           630              1  ...        0        0        0        0        0        0        0        0        0
4         0     2005           4   69367   33619    4000.0             0          1020              1  ...        0        0        0        0        0        0        0        0        0
[5 rows x 124 columns]
'''

print(dataset.info())
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 72969 entries, 0 to 72982
Columns: 124 entries, IsBadBuy to VNST_WV
dtypes: float64(1), int64(7), uint8(116)
memory usage: 15.1 MB
None
'''

# Splitting the dataset into Training and Test Set
X = dataset.drop(['IsBadBuy'], axis=1)
y = dataset['IsBadBuy']  # IsBadBuy is the target feature

# The test set is allocated 30% of the total data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)  # (51078, 123) (51078,)
print(X_test.shape, y_test.shape)  # (21891, 123) (21891,)

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
        VehYear  VehicleAge    VehOdo    VNZIP1  VehBCost  IsOnlineSale  WarrantyCost  Auction_ADESA  ...   VNST_PA   VNST_SC   VNST_TN   VNST_TX  VNST_UT   VNST_VA   VNST_WA   VNST_WV
0      0.381494   -0.106359  0.557592 -0.420274 -0.364077     -0.160324     -0.397782      -0.496553  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
1      1.537006   -1.274279 -1.403573  0.613301  0.610754     -0.160324     -0.595525      -0.496553  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
2     -0.774017    0.477602  0.458613  1.030658  0.329389     -0.160324     -0.201715      -0.496553  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
3      0.381494   -0.690319 -2.025237 -1.067722  0.809700     -0.160324     -0.503356       2.013883  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
4      0.959250   -0.690319  0.356691  0.089166  0.599386      6.237362      1.366820      -0.496553  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
...         ...         ...       ...       ...       ...           ...           ...            ...  ...       ...       ...       ...       ...      ...       ...       ...       ...
51073 -0.196261    0.477602 -0.812232  0.655439 -0.315762     -0.160324     -0.178254      -0.496553  ... -0.108841 -0.248156 -0.157279  2.080613 -0.11139 -0.152373 -0.040344 -0.062855
51074  0.959250   -1.274279  1.192467  0.231576  0.130444     -0.160324      1.469043      -0.496553  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
51075  1.537006   -1.274279  1.161048 -1.093747  1.014329     -0.160324      1.469043      -0.496553  ... -0.108841  4.029716 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
51076 -0.196261   -0.106359  1.286449 -1.351356  0.588018     -0.160324     -0.372645      -0.496553  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139  6.562831 -0.040344 -0.062855
51077  0.381494   -0.690319  0.041273  0.869035 -0.162290     -0.160324      0.386486       2.013883  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
[51078 rows x 123 columns]
'''

print(X_test_scaled)
'''
        VehYear  VehicleAge    VehOdo    VNZIP1  VehBCost  IsOnlineSale  WarrantyCost  Auction_ADESA  ...   VNST_PA   VNST_SC   VNST_TN   VNST_TX  VNST_UT   VNST_VA   VNST_WA   VNST_WV
0      0.381494   -0.690319 -1.403436  0.869035  0.269705     -0.160324      0.376431       2.013883  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
1      0.381494   -0.690319 -2.171789 -1.141931 -0.682389     -0.160324     -1.363034       2.013883  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
2     -0.196261   -0.106359 -0.891840  1.037776  1.062644     -0.160324     -0.595525       2.013883  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
3     -0.196261   -0.106359 -1.056052  0.838188 -0.210605     -0.160324     -0.595525      -0.496553  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
4      0.381494   -0.106359  0.950496 -1.087432 -0.412392     -0.160324     -0.054246      -0.496553  ... -0.108841  4.029716 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
...         ...         ...       ...       ...       ...           ...           ...            ...  ...       ...       ...       ...       ...      ...       ...       ...       ...
21886 -0.774017    0.477602 -0.140120  1.437604 -1.327540     -0.160324     -0.917275      -0.496553  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
21887  0.381494   -0.106359  0.436025  0.655439  0.082129     -0.160324     -0.397782      -0.496553  ... -0.108841 -0.248156 -0.157279  2.080613 -0.11139 -0.152373 -0.040344 -0.062855
21888  0.381494   -0.106359 -0.190157  1.327917 -0.622706     -0.160324     -0.917275      -0.496553  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
21889  0.381494   -0.106359 -0.655892 -0.716002 -0.165132     -0.160324     -0.917275      -0.496553  ... -0.108841 -0.248156 -0.157279 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
21890 -0.196261    0.477602 -1.090003 -0.778423 -0.548812     -0.160324     -0.317344       2.013883  ... -0.108841 -0.248156  6.358128 -0.480628 -0.11139 -0.152373 -0.040344 -0.062855
[21891 rows x 123 columns]
'''

'''
#모든 스케일링

#For standard scaled dataset
#Convert range to normal distribution on existing variable
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)
X_train_scaled = standard_scaler.transform(X_train)
X_test_scaled = standard_scaler.transform(X_test)

#For minmax scaled dataset
#Convert to a value between 0 and 1
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X_train)
train_img_scaled = minmax_scaler.transform(X_train)
test_img_scaled = minmax_scaler.transform(X_test)

#For maxabs scaled dataset
#Converts the number with the largest absolute value to 1 or -1 based on 0
#Similar to MinMaxScaler in feature datasets consisting only of positive data
maxabs_scaler = MaxAbsScaler()
maxabs_scaler.fit(X_train)
train_img_scaled = maxabs_scaler.transform(X_train)
test_img_scaled = maxabs_scaler.transform(X_test)

#For robust scaled dataset
#Similar to Standard Scaler, but instead of mean and variance, Median and IQR (interquartile range) are used
robust_scaler = RobustScaler()
robust_scaler.fit(X_train)
train_img_scaled = robust_scaler.transform(X_train)
test_img_scaled = robust_scaler.transform(X_test)
'''

# #Correlation Analysis
# corrmat = dataset.corr() # corr() computes pairwise correlations of features in a Data Frame
# top_corr_features = corrmat.index
# plt.figure(figsize=(20,20))
# # Plot the heat map
# g = sns.heatmap(dataset[top_corr_features].corr(), annot=True, cmap="rainbow")
# plt.show()

# Decision tree evaluation
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

accuracy = accuracy_score(y_test, pred_dt)
print('Decision Tree best estimator accuracy: {0:.4f}'.format(accuracy))
#feature의 중요도 plt로 나타내기
import seaborn as sns
feature_importance_values = best_df_clf.feature_importances_
# Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
feature_importances = pd.Series(feature_importance_values, index=X_train.columns)
# 중요도값 순으로 Series를 정렬
feature_top20 = feature_importances.sort_values(ascending=False)[:20]
feature_top1 = feature_top20.index[0]
feature_top2 = feature_top20.index[1]
print("feature top1: {0}, feature top2: {1}\n".format(feature_top1,feature_top2))
plt.figure(figsize=[8, 6])
plt.title('Feature Importances Top 20')
sns.barplot(x=feature_top20, y=feature_top20.index)
plt.show()




# KNN algorithm evaluation
from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_neighbors':[1,2,3,4,5]}
estimator = KNeighborsClassifier()
grid = GridSearchCV(estimator, param_grid=param_grid)

grid.fit(X_train_scaled, y_train)
print("K-nearest neighbors best score: {0:.4f}".format(grid.best_score_))
print("best hyperparameter:")
print(grid.best_params_)
# KNN 최적의 모델
best_df_knn = grid.best_estimator_
# 최적 모델 예측결과
pred_knn = best_df_knn.predict(X_test_scaled)
submission = pd.DataFrame(data=pred_knn,columns=["IsBadBuy"])
print("best KNN predict:")
print(submission)

accuracy = accuracy_score(y_test, pred_knn)
print('k-nearest neighbor accuracy: {0:.4f}'.format(accuracy))
print("\n")



# ensemble learning - Random Forest
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
print('Random forest best score: {:.4f}'.format(rf_grid.best_score_))
print('Random forest best parameter: ', rf_grid.best_params_)
# 최적의 파라미터값을 만든 모델
best_rf_clf = rf_grid.best_estimator_
pred_rf = best_rf_clf.predict(X_test_scaled)
# Random Forest 예측결과
print("Random Forest best estimator prediction")
submission = pd.DataFrame(data=pred_rf,columns=["IsBadBuy"])
print(submission)

accuracy = accuracy_score(y_test, pred_rf)
print('Random Forest accuracy: {0:.4f}'.format(accuracy))
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