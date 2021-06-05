--------------------------------------------------------------------------------------------------------------

# Open Source SW Contribution Project Document

--------------------------------------------------------------------------------------------------------------

Class **EasyCombination**

This class and methods automatically generate a combination of various sclaers and encoders, and calculates the scores according to each combination.

Contain Methods: encodeAndSplit, Scale, Estimate, printAllResult, findBestScore, findWorstScore

WARNING: encodeAndSplit, Scale, Estimate must be executed first before printAllResult, findBestScore, findWorstScore

_ **Constructor** _

**\_\_init\_\_:**

\_\_init\_\_(self, dataset)

**Implementation Note:**

After the dataset is loaded, this constructor can be called to use the class.

**Parameters:**

_dataset_: Dataset you will use

_ **Method Details** _

| Method | Description |
| --- | --- |
| encodeAndSplit(self, target\_name) | **Parameters** : self, target\_name
You have to deliver target\_name to estimate.
**Returns** : void
**Description** :Encodes dataset with One-Hot, Label encoder.Then split dataset into train and test.Recommend call the _scale_ method after this method call
**Examples** :\&gt;\&gt;\&gt;import opensource\&gt;\&gt;\&gt;os = opensource.EasyCombination(newDatset)
 \&gt;\&gt;\&gt;os.encodeAndSplit(&#39;IsBadBuy&#39;)
 |
| scale(self) | **Parameters** : self **Returns** : void
**Description** :Executes scaling: standard, MinMax, MaxAbs, RobustRecommend call the _estimate_ method after this method call
**Examples** :\&gt;\&gt;\&gt;import opensource\&gt;\&gt;\&gt;os = opensource.EasyCombination(newDatset)
 \&gt;\&gt;\&gt;os.encodeAndSplit(&#39;IsBadBuy&#39;)
\&gt;\&gt;\&gt;os.scale() |
| estimate(self, option) | **Parameters** : self, option
 # option 1: DecisionTree
 # option 2: KNN
 # option 3: Random Forest
 # option others: Error
**Returns** : void
**Description** :Calculates estimation scores for each combination: #Label Encoding + Standard Scale#Label Encoding + MinMax Scale#Label Encoding + MaxAbs Scale#Label Encoding + Robust Scale# OneHot Encoding + Standard Scale# OneHot Encoding + MinMax Scale# OneHot Encoding + MaxAbs Scale# OneHot Encoding + Robust Scale
**Examples** :\&gt;\&gt;\&gt;import opensource\&gt;\&gt;\&gt;os = opensource.EasyCombination(newDatset)
 \&gt;\&gt;\&gt;os.encodeAndSplit(&#39;IsBadBuy&#39;)\&gt;\&gt;\&gt;os.scale()
\&gt;\&gt;\&gt;os.estimate(1) # option 1: DecisionTree

Selected Random ForestScore using Label Encoding and Standard Scaling: 0.9009181855557078Score using Label Encoding and MinMax Scaling: 0.9002786533278516Score using Label Encoding and Robust Scaling: 0.8998675254670869Score using One-Hot and Standard Scaling: 0.9005527385683614Score using One-Hot Encoding and MinMax Scaling: 0.9008268238088712Score using One-Hot Encoding and Maxabs Scaling: 0.9007354620620346Score using One-Hot Encoding and Robust Scaling: 0.9005984194417798
 |
| printAllResult(self) | **Parameters** : self **Returns** : void
**Description** :Prints all the results of the estimate method.
**Examples** :\&gt;\&gt;\&gt;import opensource\&gt;\&gt;\&gt;os = opensource.EasyCombination(newDatset)
 \&gt;\&gt;\&gt;os.encodeAndSplit(&#39;IsBadBuy&#39;)\&gt;\&gt;\&gt;os.scale()
\&gt;\&gt;\&gt;os.estimate(1) # option 1: DecisionTree\&gt;\&gt;\&gt;os.printAllResult()


Selected Random ForestScore using Label Encoding and Standard Scaling: 0.9009181855557078Score using Label Encoding and MinMax Scaling: 0.9002786533278516Score using Label Encoding and Robust Scaling: 0.8998675254670869Score using One-Hot and Standard Scaling: 0.9005527385683614Score using One-Hot Encoding and MinMax Scaling: 0.9008268238088712Score using One-Hot Encoding and Maxabs Scaling: 0.9007354620620346Score using One-Hot Encoding and Robust Scaling: 0.9005984194417798

Best result score is: 0.9009181855557078Using Label Encoding and Standard Scaling
Worst result score is: 0.8998675254670869Using Label Encoding and Robust Scaling |
| findBestScore(self) | **Parameters** : self **Returns** : void
**Description** :Outputs the highest score of the estimation
**Examples** :\&gt;\&gt;\&gt;import opensource\&gt;\&gt;\&gt;os = opensource.EasyCombination(newDatset)
 \&gt;\&gt;\&gt;os.encodeAndSplit(&#39;IsBadBuy&#39;)\&gt;\&gt;\&gt;os.scale()\&gt;\&gt;\&gt;os.estimate(1) # option 1: DecisionTree
\&gt;\&gt;\&gt;os.findBestScore()

Best result score is: 0.9009181855557078Using Label Encoding and Standard Scaling |
| findWorstScore(self) | **Parameters** : self **Returns** : void
**Description** :Outputs the Lowest score of the estimation
**Examples** :\&gt;\&gt;\&gt;import opensource\&gt;\&gt;\&gt;os = opensource.EasyCombination(newDatset)
 \&gt;\&gt;\&gt;os.encodeAndSplit(&#39;IsBadBuy&#39;)\&gt;\&gt;\&gt;os.scale()\&gt;\&gt;\&gt;os.estimate(1) # option 1: DecisionTree
\&gt;\&gt;\&gt;os.findWorstScore()

Worst result score is: 0.8998675254670869Using Label Encoding and Robust Scaling |
