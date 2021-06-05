\--------------------------------------------------------------------------------------------------------------

Open Source SW Contribution Project Document

\--------------------------------------------------------------------------------------------------------------

Class **EasyCombination**

This class and methods automatically generate a combination of various sclaers and encoders, and calculates the scores according to each combination.

Contain Methods: encodeAndSplit, Scale, Estimate, printAllResult, findBestScore, findWorstScore

WARNING: encodeAndSplit, Scale, Estimate must be executed first before printAllResult, findBestScore, findWorstScore

***Constructor***

**\_\_init\_\_:**

\_\_init\_\_(self, dataset)

**Implementation Note:**

After the dataset is loaded, this constructor can be called to use the class.

**Parameters:** 

*dataset*: Dataset you will use




***Method Details***

|Method|Description|
| :- | :- |
|encodeAndSplit(self, target\_name)|<p>**Parameters**: self, target\_name</p><p></p><p>You have to deliver target\_name to estimate.</p><p></p><p>**Returns**: void</p><p></p><p>**Description**:</p><p>Encodes dataset with One-Hot, Label encoder.</p><p>Then split dataset into train and test.</p><p>Recommend call the *scale* method after this method call</p><p></p><p>**Examples**:</p><p>>>>import opensource</p><p>>>>os = opensource.EasyCombination(newDatset)<br>>>>os.encodeAndSplit('IsBadBuy')</p><p></p>|
|scale(self) |<p>**Parameters**: self</p><p>**Returns**: void</p><p></p><p>**Description**:</p><p>Executes scaling: standard, MinMax, MaxAbs, Robust</p><p>Recommend call the *estimate* method after this method call</p><p></p><p>**Examples**:</p><p>>>>import opensource</p><p>>>>os = opensource.EasyCombination(newDatset)<br>>>>os.encodeAndSplit('IsBadBuy')</p><p></p><p>>>>os.scale()</p>|
|estimate(self, option)|<p>**Parameters**: self, option</p><p><br># option 1: DecisionTree<br># option 2: KNN<br># option 3: Random Forest<br># option others: Error</p><p></p><p>**Returns**: void</p><p></p><p>**Description**:</p><p>Calculates estimation scores for each combination: </p><p>#Label Encoding + Standard Scale</p><p>#Label Encoding + MinMax Scale</p><p>#Label Encoding + MaxAbs Scale</p><p>#Label Encoding + Robust Scale</p><p># OneHot Encoding + Standard Scale</p><p># OneHot Encoding + MinMax Scale</p><p># OneHot Encoding + MaxAbs Scale</p><p># OneHot Encoding + Robust Scale</p><p></p><p>**Examples**:</p><p>>>>import opensource</p><p>>>>os = opensource.EasyCombination(newDatset)<br>>>>os.encodeAndSplit('IsBadBuy')</p><p>>>>os.scale()</p><p></p><p>>>>os.estimate(1) # option 1: DecisionTree</p><p></p><p></p><p>Selected Random Forest</p><p>Score using Label Encoding and Standard Scaling: 0.9009181855557078</p><p>Score using Label Encoding and MinMax Scaling: 0.9002786533278516</p><p>Score using Label Encoding and Robust Scaling: 0.8998675254670869</p><p>Score using One-Hot and Standard Scaling: 0.9005527385683614</p><p>Score using One-Hot Encoding and MinMax Scaling: 0.9008268238088712</p><p>Score using One-Hot Encoding and Maxabs Scaling: 0.9007354620620346</p><p>Score using One-Hot Encoding and Robust Scaling: 0.9005984194417798</p><p></p>|
|printAllResult(self)|<p>**Parameters**: self</p><p>**Returns**: void</p><p></p><p>**Description**:</p><p>Prints all the results of the estimate method.</p><p></p><p>**Examples**:</p><p>>>>import opensource</p><p>>>>os = opensource.EasyCombination(newDatset)<br>>>>os.encodeAndSplit('IsBadBuy')</p><p>>>>os.scale()</p><p></p><p>>>>os.estimate(1) # option 1: DecisionTree</p><p>>>>os.printAllResult()</p><p></p><p></p><p></p><p>Selected Random Forest</p><p>Score using Label Encoding and Standard Scaling: 0.9009181855557078</p><p>Score using Label Encoding and MinMax Scaling: 0.9002786533278516</p><p>Score using Label Encoding and Robust Scaling: 0.8998675254670869</p><p>Score using One-Hot and Standard Scaling: 0.9005527385683614</p><p>Score using One-Hot Encoding and MinMax Scaling: 0.9008268238088712</p><p>Score using One-Hot Encoding and Maxabs Scaling: 0.9007354620620346</p><p>Score using One-Hot Encoding and Robust Scaling: 0.9005984194417798</p><p></p><p></p><p>Best result score is: 0.9009181855557078</p><p>Using Label Encoding and Standard Scaling</p><p></p><p>Worst result score is: 0.8998675254670869</p><p>Using Label Encoding and Robust Scaling</p>|
|findBestScore(self)|<p>**Parameters**: self</p><p>**Returns**: void</p><p></p><p>**Description**:</p><p>Outputs the highest score of the estimation</p><p></p><p>**Examples**:</p><p>>>>import opensource</p><p>>>>os = opensource.EasyCombination(newDatset)<br>>>>os.encodeAndSplit('IsBadBuy')</p><p>>>>os.scale()</p><p>>>>os.estimate(1) # option 1: DecisionTree</p><p></p><p>>>>os.findBestScore()</p><p></p><p></p><p>Best result score is: 0.9009181855557078</p><p>Using Label Encoding and Standard Scaling</p>|
|findWorstScore(self)|<p>**Parameters**: self</p><p>**Returns**: void</p><p></p><p>**Description**:</p><p>Outputs the Lowest score of the estimation</p><p></p><p>**Examples**:</p><p>>>>import opensource</p><p>>>>os = opensource.EasyCombination(newDatset)<br>>>>os.encodeAndSplit('IsBadBuy')</p><p>>>>os.scale()</p><p>>>>os.estimate(1) # option 1: DecisionTree</p><p></p><p>>>>os.findWorstScore()</p><p></p><p></p><p>Worst result score is: 0.8998675254670869</p><p>Using Label Encoding and Robust Scaling</p>|

