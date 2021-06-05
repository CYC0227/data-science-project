## EasyCombination
This class and methods automatically generate a combination of various sclaers and encoders, and calculates the scores according to each combination
Contain Methods: encodeAndSplit, Scale, Estimate, printAllResult, findBestScore, findWorstScore
WARNING: encodeAndSplit, Scale, Estimate must be executed first before printAllResult, findBestScore, findWorstScore


## Constructor
__init__:
__init__(self, dataset)

Implementation Note:
After the dataset is loaded, this constructor can be called to use the class.

Parameters: 
dataset: Dataset you will use

## Method Details

|     Method       |      Description   | 
| :--------------- | :----------------: | 
| encodeAndSplit(self, target_name)|Parameters: self, target_name

You have to deliver target_name to estimate.

Returns: void

Description:
Encodes dataset with One-Hot, Label encoder.
Then split dataset into train and test.
Recommend call the scale method after this method call

Examples:
>>>import opensource
>>>os = opensource.EasyCombination(newDatset)
>>>os.encodeAndSplit('IsBadBuy')
  | 
| state            | :white_check_mark: | 
| isAvailable      | :white_check_mark: |  
| isOn             | :white_check_mark: |
