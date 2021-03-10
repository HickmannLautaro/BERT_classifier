[//]: # (TODO REDO)
# BERT for hierarchicla text classification on a dataset with Amazon product reviews
Using the pretrained bert-base-uncased form [Hugging Face](https://huggingface.co/bert-base-uncased)   

Dataset with Amazon product reviews, classes are structured as:
* 6 "level 1" classes
* 64 "level 2" classes
* 510 "level 3" classes

The training set contains 40k documents and the test set 10k.  
Each document contains: Title, Text, Cat1, Cat2, Cat3  
## Model architectures

### Hierarchical
![hierarchical](./visualizations/hierarchical.png)


## Newest results
## Only predicted results
In all cases as average of $3$ runs for $100$ Tokens

| Dataset   | Type      | Epochs   | Batch size| Train Input                  | Output   | Test Input                      | Cat1 accuracy   | Cat1 F1 score macro   | Cat2 accuracy   | Cat2 F1 score macro   | Cat3 accuracy   | Cat3 F1 score macro   |
|:----------|:----------|:---------|:----------|:-----------------------------|:---------|:--------------------------------|:----------------|:----------------------|:----------------|:----------------------|:----------------|:----------------------|
| amazon    | Per_lvl   | 60(10)   | 45        | Text                         | Cat1     | Text                            | 0.828(0.002)    | 0.819(0.002)          | -               | -                     | -               | -                     |
| amazon    | Per_lvl   | 60(53)   | 40        | Text                         | Cat2     | Text                            | -               | -                     | 0.622(0.003)    | 0.393(0.004)          | -               | -                     |
| amazon    | Per_lvl   | 60(60)   | 45        | Target Cat1, Text            | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.612(0.000)    | 0.400(0.002)          | -               | -                     |
| amazon    | Per_lvl   | 60(60)   | 45        | Predicted Cat1, Text         | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.624(0.002)    | 0.398(0.002)          | -               | -                     |
| amazon    | Per_label | 60       | 45        | Text divided per Target Cat1 | Cat2     | Text divided per Predicted Cat1 | -               | -                     | 0.624(0.000)    | 0.474(0.005)          | -               | -                     |
| ---       | --------- | ---      | ---       | ---                          | ---      | ---                             | ---             | ---                   | ---             | ---                   | ---             | ---                   |
| dbpedia   | Per_lvl   | 20(20)   | 40        | Text                         | Cat1     | Text                            | 0.996(0.000)    | 0.994(0.000)          | -               | -                     | -               | -                     |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Text                         | Cat2     | Text                            | -               | -                     | 0.975(0.000)    | 0.968(0.001)          | -               | -                     |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Target Cat1, Text            | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.976(0.000)    | 0.970(0.001)          | -               | -                     |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Predicted Cat1, Text         | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.976(0.000)    | 0.969(0.001)          | -               | -                     |
| dbpedia   | Per_label | 60       | 45        | Text divided per Target Cat1 | Cat2     | Text divided per Predicted Cat1 | -               | -                     | 0.977(0.000)    | 0.972(0.001)          | -               | -                     |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Text                         | Cat3     | Text                            | -               | -                     | -               | -                     | 0.951(0.001)    | 0.918(0.007)          |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Target Cat1, Cat2, Text      | Cat3     | Predicted Cat1, Cat2, Text      | -               | -                     | -               | -                     | 0.954(0.001)    | 0.931(0.003)          |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Predicted Cat1, Cat2, Text   | Cat3     | Predicted Cat1, Cat2, Text      | -               | -                     | -               | -                     | 0.953(0.001)    | 0.922(0.005)          |
| dbpedia   | Per_label | 60       | 45        | Text divided per Target Cat2 | Cat3     | Text divided per Predicted Cat2 | -               | -                     | -               | -                     | 0.962(0.000)    | 0.957(0.000)          |


| Dataset   | Type      | Epochs   | Batch size| Train Input                  | Output   | Test Input                      | Cat accuracy    | Cat F1 score macro    |
|:----------|:----------|:---------|:----------|:-----------------------------|:---------|:--------------------------------|:----------------|:----------------------|
| amazon    | Per_lvl   | 60(10)   | 45        | Text                         | Cat1     | Text                            | 0.828(0.002)    | 0.819(0.002)          |
| amazon    | Per_lvl   | 60(53)   | 40        | Text                         | Cat2     | Text                            | 0.622(0.003)    | 0.393(0.004)          |
| amazon    | Per_lvl   | 60(60)   | 45        | Target Cat1, Text            | Cat2     | Predicted Cat1, Text            | 0.612(0.000)    | 0.400(0.002)          |
| amazon    | Per_lvl   | 60(60)   | 45        | Predicted Cat1, Text         | Cat2     | Predicted Cat1, Text            | 0.624(0.002)    | 0.398(0.002)          |
| amazon    | Per_label | 60       | 45        | Text divided per Target Cat1 | Cat2     | Text divided per Predicted Cat1 | 0.624(0.000)    | 0.474(0.005)          |
| ---       | --------- | ---      | ---       | ---                          | ---      | ---                             | ---             | ---                   |
| dbpedia   | Per_lvl   | 20(20)   | 40        | Text                         | Cat1     | Text                            | 0.996(0.000)    | 0.994(0.000)          |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Text                         | Cat2     | Text                            | 0.975(0.000)    | 0.968(0.001)          |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Target Cat1, Text            | Cat2     | Predicted Cat1, Text            | 0.976(0.000)    | 0.970(0.001)          |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Predicted Cat1, Text         | Cat2     | Predicted Cat1, Text            | 0.976(0.000)    | 0.969(0.001)          |
| dbpedia   | Per_label | 60       | 45        | Text divided per Target Cat1 | Cat2     | Text divided per Predicted Cat1 | 0.977(0.000)    | 0.972(0.001)          |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Text                         | Cat3     | Text                            | 0.951(0.001)    | 0.918(0.007)          |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Target Cat1, Cat2, Text      | Cat3     | Predicted Cat1, Cat2, Text      | 0.954(0.001)    | 0.931(0.003)          |
| dbpedia   | Per_lvl   | 40(40)   | 40        | Predicted Cat1, Cat2, Text   | Cat3     | Predicted Cat1, Cat2, Text      | 0.953(0.001)    | 0.922(0.005)          |
| dbpedia   | Per_label | 60       | 45        | Text divided per Target Cat2 | Cat3     | Text divided per Predicted Cat2 | 0.962(0.000)    | 0.957(0.000)          |


## predicted results with target target for comparisson
In all cases as average of $3$ runs for $100$ Tokens

| Type      | Epochs   | Batch size | Train Input **amazon**       | Output **amazon** | Test Input               | Cat1 accuracy   | Cat1 F1 score macro   |
|:----------|:---------|:-----------|:-----------------------------|:--------------|:--------------------------------|:----------------|:----------------------|
| Per_lvl   | 60(10)   | 45         | Text                         | Cat1          | Text                            | 0.828(0.002)    | 0.819(0.002)          |
| ---       | ---      | ---        | ---                          | ---           | ---                             |**Cat2 accuracy**|**Cat2 F1 score macro**|
| Per_lvl   | 60(53)   | 40         | Text                         | Cat2          | Text                            | 0.622(0.003)    | 0.393(0.004)          |
| Per_lvl   | 60(60)   | 45         | Target Cat1, Text            | Cat2          | Target Cat1, Text               | 0.706(0.001)    | 0.464(0.004)          |
| Per_lvl   | 60(60)   | 45         | Target Cat1, Text            | Cat2          | Predicted Cat1, Text            | 0.612(0.000)    | 0.400(0.002)          |
| Per_lvl   | 60(60)   | 45         | Predicted Cat1, Text         | Cat2          | Predicted Cat1, Text            | 0.624(0.002)    | 0.398(0.002)          |
| Per_label | 60       | 45         | Text divided per Target Cat1 | Cat2          | Text divided per Target Cat1    | 0.722(0.001)    | 0.567(0.004)          |
| Per_label | 60       | 45         | Text divided per Target Cat1 | Cat2          | Text divided per Predicted Cat1 | 0.624(0.000)    | 0.474(0.005)          |
|=======    |======    |========    |=====================         |============   |=======================          |===========      |=================      | 
| ===       | ===      | ===        | Train Input **dbpedia**      | Output **dbpedia** | Test Input **dbpedia**     |**Cat1 accuracy**|**Cat1 F1 score macro**|
| Per_lvl   | 20(20)   | 40         | Text                         | Cat1          | Text                            | 0.996(0.000)    | 0.994(0.000)          |
| ---       | ---      | ---        | ---                          | ---           | ---                             |**Cat2 accuracy**|**Cat2 F1 score macro**|
| Per_lvl   | 40(40)   | 40         | Text                         | Cat2          | Text                            | 0.975(0.000)    | 0.968(0.001)          |
| Per_lvl   | 40(40)   | 40         | Target Cat1, Text            | Cat2          | Target Cat1, Text               | 0.979(0.000)    | 0.975(0.001)          |
| Per_lvl   | 40(40)   | 40         | Target Cat1, Text            | Cat2          | Predicted Cat1, Text            | 0.976(0.000)    | 0.970(0.001)          |
| Per_lvl   | 40(40)   | 40         | Predicted Cat1, Text         | Cat2          | Predicted Cat1, Text            | 0.976(0.000)    | 0.969(0.001)          |
| Per_label | 60       | 45         | Text divided per Target Cat1 | Cat2          | Text divided per Target Cat1    | 0.980(0.000)    | 0.977(0.001)          |
| Per_label | 60       | 45         | Text divided per Target Cat1 | Cat2          | Text divided per Predicted Cat1 | 0.977(0.000)    | 0.972(0.001)          |
| ---       | ---      | ---        | ---                          | ---           | ---                             |**Cat3 accuracy**|**Cat3 F1 score macro**|
| Per_lvl   | 40(40)   | 40         | Text                         | Cat3          | Text                            | 0.951(0.001)    | 0.918(0.007)          |
| Per_lvl   | 40(40)   | 40         | Target Cat1, Cat2, Text      | Cat3          | Target Cat1, Cat2, Text         | 0.975(0.001)    | 0.954(0.003)          |
| Per_lvl   | 40(40)   | 40         | Target Cat1, Cat2, Text      | Cat3          | Predicted Cat1, Cat2, Text      | 0.954(0.001)    | 0.931(0.003)          |
| Per_lvl   | 40(40)   | 40         | Predicted Cat1, Cat2, Text   | Cat3          | Predicted Cat1, Cat2, Text      | 0.953(0.001)    | 0.922(0.005)          |
| Per_label | 60       | 45         | Text divided per Target Cat2 | Cat3          | Text divided per Target Cat2    | 0.983(0.000)    | 0.982(0.000)          |
| Per_label | 60       | 45         | Text divided per Target Cat2 | Cat3          | Text divided per Predicted Cat2 | 0.962(0.000)    | 0.957(0.000)          |






| Type      | Dataset   | Epochs   | Tokens   | Batch size   | Runs   | Train Input                  | Output   | Test Input                      | Cat1 accuracy   | Cat1 F1 score macro   | Cat2 accuracy   | Cat2 F1 score macro   | Cat3 accuracy   | Cat3 F1 score macro   |
|:----------|:----------|:---------|:---------|:-------------|:-------|:-----------------------------|:---------|:--------------------------------|:----------------|:----------------------|:----------------|:----------------------|:----------------|:----------------------|
| Per_lvl   | amazon    | 60(10)   | 100      | 45           | 3      | Text                         | Cat1     | Text                            | 0.828(0.002)    | 0.819(0.002)          | -               | -                     | -               | -                     |
| ---       | ---       | ---      | ---      | ---          | ---    | ---                          | ---      | ---                             | ---             | ---                   | ---             | ---                   | ---             | ---                   |
| Per_lvl   | amazon    | 60(53)   | 100      | 40           | 3      | Text                         | Cat2     | Text                            | -               | -                     | 0.622(0.003)    | 0.393(0.004)          | -               | -                     |
| Per_lvl   | amazon    | 60(60)   | 100      | 45           | 3      | Target Cat1, Text            | Cat2     | Target Cat1, Text               | -               | -                     | 0.706(0.001)    | 0.464(0.004)          | -               | -                     |
| Per_lvl   | amazon    | 60(60)   | 100      | 45           | 3      | Target Cat1, Text            | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.612(0.000)    | 0.400(0.002)          | -               | -                     |
| Per_lvl   | amazon    | 60(60)   | 100      | 45           | 3      | Predicted Cat1, Text         | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.624(0.002)    | 0.398(0.002)          | -               | -                     |
| Per_label | amazon    | 60       | 100      | 45           | 3      | Text divided per Target Cat1 | Cat2     | Text divided per Target Cat1    | -               | -                     | 0.722(0.001)    | 0.567(0.004)          | -               | -                     |
| Per_label | amazon    | 60       | 100      | 45           | 3      | Text divided per Target Cat1 | Cat2     | Text divided per Predicted Cat1 | -               | -                     | 0.624(0.000)    | 0.474(0.005)          | -               | -                     |
| ===       | ===       | ===      | ===      | ===          | ===    | ===                          | ===      | ===                             | ===             | ===                   | ---             | ---                   | ---             | ---                   |
| Per_lvl   | dbpedia   | 20(20)   | 100      | 40           | 3      | Text                         | Cat1     | Text                            | 0.996(0.000)    | 0.994(0.000)          | -               | -                     | -               | -                     |
| ---       | ---       | ---      | ---      | ---          | ---    | ---                          | ---      | ---                             | ---             | ---                   | ---             | ---                   | ---             | ---                   |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Text                         | Cat2     | Text                            | -               | -                     | 0.975(0.000)    | 0.968(0.001)          | -               | -                     |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Target Cat1, Text            | Cat2     | Target Cat1, Text               | -               | -                     | 0.979(0.000)    | 0.975(0.001)          | -               | -                     |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Target Cat1, Text            | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.976(0.000)    | 0.970(0.001)          | -               | -                     |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Predicted Cat1, Text         | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.976(0.000)    | 0.969(0.001)          | -               | -                     |
| Per_label | dbpedia   | 60       | 100      | 45           | 3      | Text divided per Target Cat1 | Cat2     | Text divided per Target Cat1    | -               | -                     | 0.980(0.000)    | 0.977(0.001)          | -               | -                     |
| Per_label | dbpedia   | 60       | 100      | 45           | 3      | Text divided per Target Cat1 | Cat2     | Text divided per Predicted Cat1 | -               | -                     | 0.977(0.000)    | 0.972(0.001)          | -               | -                     |
| ---       | ---       | ---      | ---      | ---          | ---    | ---                          | ---      | ---                             | ---             | ---                   | ---             | ---                   | ---             | ---                   |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Text                         | Cat3     | Text                            | -               | -                     | -               | -                     | 0.951(0.001)    | 0.918(0.007)          |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Target Cat1, Cat2, Text      | Cat3     | Target Cat1, Cat2, Text         | -               | -                     | -               | -                     | 0.975(0.001)    | 0.954(0.003)          |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Target Cat1, Cat2, Text      | Cat3     | Predicted Cat1, Cat2, Text      | -               | -                     | -               | -                     | 0.954(0.001)    | 0.931(0.003)          |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Predicted Cat1, Cat2, Text   | Cat3     | Predicted Cat1, Cat2, Text      | -               | -                     | -               | -                     | 0.953(0.001)    | 0.922(0.005)          |
| Per_label | dbpedia   | 60       | 100      | 45           | 3      | Text divided per Target Cat2 | Cat3     | Text divided per Target Cat2    | -               | -                     | -               | -                     | 0.983(0.000)    | 0.982(0.000)          |
| Per_label | dbpedia   | 60       | 100      | 45           | 3      | Text divided per Target Cat2 | Cat3     | Text divided per Predicted Cat2 | -               | -                     | -               | -                     | 0.962(0.000)    | 0.957(0.000)          |






## Check which experiments make sense

|    | Dataset | Train Input            | Output | Test Input                | Useful |
|----|---------|------------------------|--------|---------------------------|--------|
| 0  | amazon  | Text                   | Cat1   | Text                      | x      |
| 1  | amazon  | Text                   | Cat2   | Text                      | x      |
| 2  | amazon  | Text                   | Cat2   | Target Cat1, Text         |        |
| 3  | amazon  | Text                   | Cat2   | Predicted Cat1, Text      |        |
| 4  | amazon  | Target Cat1, Text      | Cat2   | Text                      |        |
| 5  | amazon  | Target Cat1, Text      | Cat2   | Target Cat1, Text         | x      |
| 6  | amazon  | Target Cat1, Text      | Cat2   | Predicted Cat1, Text      | x      |
| 7  | amazon  | Predicted Cat1, Text   | Cat2   | Text                      |        |
| 8  | amazon  | Predicted Cat1, Text   | Cat2   | Target Cat1, Text         |        |
| 9  | amazon  | Predicted Cat1, Text   | Cat2   | Predicted Cat1, Text      | x  ?   |
| -  | ------- | ---------------------- | ------ | ------------------------- | ------ |
| 10 | dbpedia | Text                   | Cat1   | Text                      | x      |
| 11 | dbpedia | Text                   | Cat2   | Text                      | x      |
| 12 | dbpedia | Text                   | Cat2   | Target Cat1, Text         |        |
| 13 | dbpedia | Text                   | Cat2   | Predicted Cat1, Text      |        |
| 14 | dbpedia | Target Cat1, Text      | Cat2   | Text                      |        |
| 15 | dbpedia | Target Cat1, Text      | Cat2   | Target Cat1, Text         | x      |
| 16 | dbpedia | Target Cat1, Text      | Cat2   | Predicted Cat1, Text      | x      |
| 17 | dbpedia | Predicted Cat1, Text   | Cat2   | Text                      |        |
| 18 | dbpedia | Predicted Cat1, Text   | Cat2   | Target Cat1, Text         |        |
| 19 | dbpedia | Predicted Cat1, Text   | Cat2   | Predicted Cat1, Text      | x      |
| 20 | dbpedia | Text                   | Cat3   | Text                      | x      |
| 21 | dbpedia | Target Cat2, Text      | Cat3   | Text                      |        |
| 22 | dbpedia | Target Cat2, Text      | Cat3   | Target Cat2, Text         |        |
| 23 | dbpedia | Target Cat1, Text      | Cat3   | Text                      |        |
| 24 | dbpedia | Target Cat1, Text      | Cat3   | Target Cat1, Text         |        |
| 25 | dbpedia | Target Cat1 Cat2, Text | Cat3   | Text                      |        |
| 26 | dbpedia | Target Cat1 Cat2, Text | Cat3   | Target Cat1 Cat2, Text    | x      |
| 27 | dbpedia | Target Cat1 Cat2, Text | Cat3   | Predicted Cat1 Cat2, Text | x      |

## New versionLatest (Filtered) Evaluation results 
Results are on the Test set

| Type      | Dataset   | Epochs   | Tokens   | Batch size   | Runs   | Train Input                  | Output   | Test Input                      | Cat1 accuracy   | Cat1 F1 score macro   | Cat2 accuracy   | Cat2 F1 score macro   | Cat3 accuracy   | Cat3 F1 score macro   |
|:----------|:----------|:---------|:---------|:-------------|:-------|:-----------------------------|:---------|:--------------------------------|:----------------|:----------------------|:----------------|:----------------------|:----------------|:----------------------|
| Per_lvl   | amazon    | 60(10)   | 100      | 45           | 3      | Text                         | Cat1     | Text                            | 0.828(0.002)    | 0.819(0.002)          | -               | -                     | -               | -                     |
| Per_label | amazon    | 60       | 100      | 45           | 3      | Text divided per Target Cat1 | Cat2     | Text divided per Target Cat1    | -               | -                     | 0.722(0.001)    | 0.567(0.004)          | -               | -                     |
| Per_label | amazon    | 60       | 100      | 45           | 3      | Text divided per Target Cat1 | Cat2     | Text divided per Predicted Cat1 | -               | -                     | 0.624(0.000)    | 0.474(0.005)          | -               | -                     |
| Per_lvl   | amazon    | 60(53)   | 100      | 40           | 3      | Text                         | Cat2     | Text                            | -               | -                     | 0.622(0.003)    | 0.393(0.004)          | -               | -                     |
| Per_lvl   | amazon    | 60(60)   | 100      | 45           | 3      | Target Cat1, Text            | Cat2     | Target Cat1, Text               | -               | -                     | 0.706(0.001)    | 0.464(0.004)          | -               | -                     |
| Per_lvl   | amazon    | 60(60)   | 100      | 45           | 3      | Target Cat1, Text            | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.612(0.000)    | 0.400(0.002)          | -               | -                     |
| Per_lvl   | amazon    | 60(60)   | 100      | 45           | 3      | Predicted Cat1, Text         | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.624(0.002)    | 0.398(0.002)          | -               | -                     |
| ---       | ---       | ---      | ---      | ---          | ---    | ---                          | ---      | ---                             | ---             | ---                   | ---             | ---                   | ---             | ---                   |
| Per_lvl   | dbpedia   | 20(20)   | 100      | 40           | 3      | Text                         | Cat1     | Text                            | 0.996(0.000)    | 0.994(0.000)          | -               | -                     | -               | -                     |
| Per_label | dbpedia   | 60       | 100      | 45           | 3      | Text divided per Target Cat1 | Cat2     | Text divided per Target Cat1    | -               | -                     | 0.980(0.000)    | 0.977(0.001)          | -               | -                     |
| Per_label | dbpedia   | 60       | 100      | 45           | 3      | Text divided per Target Cat1 | Cat2     | Text divided per Predicted Cat1 | -               | -                     | 0.977(0.000)    | 0.972(0.001)          | -               | -                     |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Text                         | Cat2     | Text                            | -               | -                     | 0.975(0.000)    | 0.968(0.001)          | -               | -                     |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Target Cat1, Text            | Cat2     | Target Cat1, Text               | -               | -                     | 0.979(0.000)    | 0.975(0.001)          | -               | -                     |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Target Cat1, Text            | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.976(0.000)    | 0.970(0.001)          | -               | -                     |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Predicted Cat1, Text         | Cat2     | Predicted Cat1, Text            | -               | -                     | 0.976(0.000)    | 0.969(0.001)          | -               | -                     |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Text                         | Cat3     | Text                            | -               | -                     | -               | -                     | 0.951(0.001)    | 0.918(0.007)          |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Target Cat1, Cat2, Text      | Cat3     | Target Cat1, Cat2, Text         | -               | -                     | -               | -                     | 0.975(0.001)    | 0.954(0.003)          |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Target Cat1, Cat2, Text      | Cat3     | Predicted Cat1, Cat2, Text      | -               | -                     | -               | -                     | 0.954(0.001)    | 0.931(0.003)          |
| Per_lvl   | dbpedia   | 40(40)   | 100      | 40           | 3      | Predicted Cat1, Cat2, Text   | Cat3     | Predicted Cat1, Cat2, Text      | -               | -                     | -               | -                     | 0.953(0.001)    | 0.922(0.005)          |




## Old Latest (Filtered) Evaluation results 
Results are on the Test set

| Config                                                    | Dataset   |   Epochs |   Tokens |   Runs | Train Input               | Output   | Test Input                | Cat1 accuracy   | Cat1 F1 score macro   | Cat2 accuracy   | Cat2 F1 score macro   | Cat3 accuracy   | Cat3 F1 score macro   |
|:----------------------------------------------------------|:----------|---------:|---------:|-------:|:--------------------------|:---------|:--------------------------|:----------------|:----------------------|:----------------|:----------------------|:----------------|:----------------------|
| amazon_config_lvl1_bert-base-uncased                      | amazon    |       20 |      100 |      2 | Text                      | Cat1     | Text                      | 0.8264          | 0.8175                | -               | -                     | -               | -                     |
| amazon_config_lvl2_flat_flatt_bert-base-uncased           | amazon    |       40 |      100 |      2 | Text                      | Cat2     | Text                      | -               | -                     | 0.6198          | 0.4025                | -               | -                     |
| amazon_config_lvl2_h_t_target_bert-base-uncased           | amazon    |       40 |      100 |      2 | Target Cat1, Text         | Cat2     | Target Cat1, Text         | -               | -                     | 0.7086          | 0.4612                | -               | -                     |
| amazon_config_lvl2_h_t_bert-base-uncased                  | amazon    |       40 |      100 |      2 | Target Cat1, Text         | Cat2     | Predicted Cat1, Text      | -               | -                     | 0.6126          | 0.3958                | -               | -                     |
| amazon_config_lvl2_h_p_bert-base-uncased                  | amazon    |       40 |      100 |      2 | Predicted Cat1, Text      | Cat2     | Predicted Cat1, Text      | -               | -                     | 0.6107          | 0.4049                | -               | -                     |
|-------------------------------------------------- | ----------|--------- |--------- |------- | ---------------------| ---------| ---------------------| ----------------| ----------------------| ----------------| ----------------------| ----------------| ----------------------|
| dbpedia_config_lvl1_bert-base-uncased                     | dbpedia   |       20 |      100 |      2 | Text                      | Cat1     | Text                      | 0.9961          | 0.9942                | -               | -                     | -               | -                     |
| dbpedia_config_lvl2_flat_flatt_bert-base-uncased          | dbpedia   |       40 |      100 |      2 | Text                      | Cat2     | Text                      | -               | -                     | 0.9751          | 0.9684                | -               | -                     |
| dbpedia_config_lvl2_h_t_target_bert-base-uncased          | dbpedia   |       40 |      100 |      2 | Target Cat1, Text         | Cat2     | Target Cat1, Text         | -               | -                     | 0.9792          | 0.9749                | -               | -                     |
| dbpedia_config_lvl2_h_t_bert-base-uncased                 | dbpedia   |       40 |      100 |      2 | Target Cat1, Text         | Cat2     | Predicted Cat1, Text      | -               | -                     | 0.9757          | 0.9697                | -               | -                     |
| dbpedia_config_lvl2_h_p_bert-base-uncased                 | dbpedia   |       40 |      100 |      2 | Predicted Cat1, Text      | Cat2     | Predicted Cat1, Text      | -               | -                     | 0.9761          | 0.9685                | -               | -                     |
| dbpedia_config_lvl3_flat_flatt_bert-base-uncased          | dbpedia   |       40 |      100 |      2 | Text                      | Cat3     | Text                      | -               | -                     | -               | -                     | 0.9517          | 0.9154                |
| dbpedia_config_lvl3_targets_targets_bert-base-uncased     | dbpedia   |       40 |      100 |      2 | Target Cat1 Cat2, Text    | Cat3     | Target Cat1 Cat2, Text    | -               | -                     | -               | -                     | 0.9742          | 0.9508                |
| dbpedia_config_lvl3_targets_predicted_bert-base-uncased   | dbpedia   |       40 |      100 |      2 | Target Cat1 Cat2, Text    | Cat3     | Predicted Cat1 Cat2, Text | -               | -                     | -               | -                     | 0.9544          | 0.9327                |
| dbpedia_config_lvl3_predicted_predicted_bert-base-uncased | dbpedia   |       40 |      100 |      2 | Predicted Cat1 Cat2, Text | Cat3     | Predicted Cat1 Cat2, Text | -               | -                     | -               | -                     | 0.9524          | 0.9207                |




## Latest Evaluation results 
Results are on the Test set

| Config                                                    | Dataset   |   Epochs |   Tokens |   Runs | Train Input               | Output   | Test Input                | Cat1 accuracy   | Cat1 F1 score macro   | Cat2 accuracy   | Cat2 F1 score macro   | Cat3 accuracy   | Cat3 F1 score macro   |
|:----------------------------------------------------------|:----------|---------:|---------:|-------:|:--------------------------|:---------|:--------------------------|:----------------|:----------------------|:----------------|:----------------------|:----------------|:----------------------|
| amazon_config_lvl1_bert-base-uncased                      | amazon    |       20 |      100 |      2 | Text                      | Cat1     | Text                      | 0.8264          | 0.8175                | -               | -                     | -               | -                     |
| amazon_config_lvl2_flat_flatt_bert-base-uncased           | amazon    |       40 |      100 |      2 | Text                      | Cat2     | Text                      | -               | -                     | 0.6198          | 0.4025                | -               | -                     |
| amazon_config_lvl2_flat_target_bert-base-uncased          | amazon    |       40 |      100 |      2 | Text                      | Cat2     | Target Cat1, Text         | -               | -                     | 0.6202          | 0.389                 | -               | -                     |
| amazon_config_lvl2_flat_bert-base-uncased                 | amazon    |       40 |      100 |      2 | Text                      | Cat2     | Predicted Cat1, Text      | -               | -                     | 0.5771          | 0.3621                | -               | -                     |
| amazon_config_lvl2_h_t_flatt_bert-base-uncased            | amazon    |       40 |      100 |      2 | Target Cat1, Text         | Cat2     | Text                      | -               | -                     | 0.4516          | 0.2664                | -               | -                     |
| amazon_config_lvl2_h_t_target_bert-base-uncased           | amazon    |       40 |      100 |      2 | Target Cat1, Text         | Cat2     | Target Cat1, Text         | -               | -                     | 0.7086          | 0.4612                | -               | -                     |
| amazon_config_lvl2_h_t_bert-base-uncased                  | amazon    |       40 |      100 |      2 | Target Cat1, Text         | Cat2     | Predicted Cat1, Text      | -               | -                     | 0.6126          | 0.3958                | -               | -                     |
| amazon_config_lvl2_h_p_flatt_bert-base-uncased            | amazon    |       40 |      100 |      2 | Predicted Cat1, Text      | Cat2     | Text                      | -               | -                     | 0.5248          | 0.311                 | -               | -                     |
| amazon_config_lvl2_h_p_Target_bert-base-uncased           | amazon    |       40 |      100 |      2 | Predicted Cat1, Text      | Cat2     | Target Cat1, Text         | -               | -                     | 0.6978          | 0.4608                | -               | -                     |
| amazon_config_lvl2_h_p_bert-base-uncased                  | amazon    |       40 |      100 |      2 | Predicted Cat1, Text      | Cat2     | Predicted Cat1, Text      | -               | -                     | 0.6107          | 0.4049                | -               | -                     |
|-----------------------------------------------------------|-----------|--------|--------|------|------------------------|--------|---------------------------|---------------|---------------------|---------------|---------------------|---------------|---------------------|
| dbpedia_config_lvl1_bert-base-uncased                     | dbpedia   |       20 |      100 |      2 | Text                      | Cat1     | Text                      | 0.9961          | 0.9942                | -               | -                     | -               | -                     |
| dbpedia_config_lvl2_flat_flatt_bert-base-uncased          | dbpedia   |       40 |      100 |      2 | Text                      | Cat2     | Text                      | -               | -                     | 0.9751          | 0.9684                | -               | -                     |
| dbpedia_config_lvl2_flat_target_bert-base-uncased         | dbpedia   |       40 |      100 |      2 | Text                      | Cat2     | Target Cat1, Text         | -               | -                     | 0.9752          | 0.968                 | -               | -                     |
| dbpedia_config_lvl2_flat_bert-base-uncased                | dbpedia   |       40 |      100 |      2 | Text                      | Cat2     | Predicted Cat1, Text      | -               | -                     | 0.9755          | 0.9692                | -               | -                     |
| dbpedia_config_lvl2_h_t_flatt_bert-base-uncased           | dbpedia   |       40 |      100 |      2 | Target Cat1, Text         | Cat2     | Text                      | -               | -                     | 0.9676          | 0.9486                | -               | -                     |
| dbpedia_config_lvl2_h_t_target_bert-base-uncased          | dbpedia   |       40 |      100 |      2 | Target Cat1, Text         | Cat2     | Target Cat1, Text         | -               | -                     | 0.9792          | 0.9749                | -               | -                     |
| dbpedia_config_lvl2_h_t_bert-base-uncased                 | dbpedia   |       40 |      100 |      2 | Target Cat1, Text         | Cat2     | Predicted Cat1, Text      | -               | -                     | 0.9757          | 0.9697                | -               | -                     |
| dbpedia_config_lvl2_h_p_flatt_bert-base-uncased           | dbpedia   |       40 |      100 |      2 | Predicted Cat1, Text      | Cat2     | Text                      | -               | -                     | 0.9561          | 0.9434                | -               | -                     |
| dbpedia_config_lvl2_h_p_Target_bert-base-uncased          | dbpedia   |       40 |      100 |      2 | Predicted Cat1, Text      | Cat2     | Target Cat1, Text         | -               | -                     | 0.979           | 0.9742                | -               | -                     |
| dbpedia_config_lvl2_h_p_bert-base-uncased                 | dbpedia   |       40 |      100 |      2 | Predicted Cat1, Text      | Cat2     | Predicted Cat1, Text      | -               | -                     | 0.9761          | 0.9685                | -               | -                     |
| dbpedia_config_lvl3_flat_flatt_bert-base-uncased          | dbpedia   |       40 |      100 |      2 | Text                      | Cat3     | Text                      | -               | -                     | -               | -                     | 0.9517          | 0.9154                |
| dbpedia_config_lvl3_2_target_flatt_bert-base-uncased      | dbpedia   |       40 |      100 |      2 | Target Cat2, Text         | Cat3     | Text                      | -               | -                     | -               | -                     | -               | -                     |
| dbpedia_config_lvl3_2_target_2_target_bert-base-uncased   | dbpedia   |       40 |      100 |      2 | Target Cat2, Text         | Cat3     | Target Cat2, Text         | -               | -                     | -               | -                     | 0.9742          | 0.9508                |
| dbpedia_config_lvl3_1_target_flatt_bert-base-uncased      | dbpedia   |       40 |      100 |      2 | Target Cat1, Text         | Cat3     | Text                      | -               | -                     | -               | -                     | -               | -                     |
| dbpedia_config_lvl3_1_target_1_target_bert-base-uncased   | dbpedia   |       40 |      100 |      2 | Target Cat1, Text         | Cat3     | Target Cat1, Text         | -               | -                     | -               | -                     | 0.9742          | 0.9508                |
| dbpedia_config_lvl3_targets_flatt_bert-base-uncased       | dbpedia   |       40 |      100 |      2 | Target Cat1 Cat2, Text    | Cat3     | Text                      | -               | -                     | -               | -                     | -               | -                     |
| dbpedia_config_lvl3_targets_targets_bert-base-uncased     | dbpedia   |       40 |      100 |      2 | Target Cat1 Cat2, Text    | Cat3     | Target Cat1 Cat2, Text    | -               | -                     | -               | -                     | 0.9742          | 0.9508                |
| dbpedia_config_lvl3_targets_predicted_bert-base-uncased   | dbpedia   |       40 |      100 |      2 | Target Cat1 Cat2, Text    | Cat3     | Predicted Cat1 Cat2, Text | -               | -                     | -               | -                     | 0.9544          | 0.9327                |
| dbpedia_config_lvl3_predicted_predicted_bert-base-uncased | dbpedia   |       40 |      100 |      2 | Predicted Cat1 Cat2, Text | Cat3     | Predicted Cat1 Cat2, Text | -               | -                     | -               | -                     | 0.9524          | 0.9207                |

## Results hyperparameter search:
Results are on the Test set
### Ordered after best f1 score and accuracy
|    | model              |   token |   epochs |     f1 |    acc |   time per epoch (m) |   f1 per second |
|---:|:-------------------|--------:|---------:|-------:|-------:|---------------------:|----------------:|
|  4 | bert-large-uncased |     200 |       10 | 0.8283 | 0.8364 |                20.35 |          0.0678 |
|  6 | bert-large-uncased |     100 |       10 | 0.8282 | 0.8364 |                10.83 |          0.1274 |
|  0 | bert-large-uncased |     100 |       20 | 0.828  | 0.8364 |                11.45 |          0.1205 |
|  3 | bert-large-uncased |     300 |       10 | 0.823  | 0.8325 |                38.9  |          0.0353 |
|  5 | bert-large-uncased |     300 |       05 | 0.823  | 0.8311 |                34.47 |          0.0398 |
|  1 | bert-large-uncased |     050 |       10 | 0.8178 | 0.828  |                 5.33 |          0.2556 |
| 10 | bert-base-uncased  |     100 |       20 | 0.8164 | 0.8257 |                 3.15 |          0.432  |
| 13 | bert-base-uncased  |     300 |       05 | 0.8148 | 0.8238 |                10.08 |          0.1347 |
| 12 | bert-base-uncased  |     100 |       05 | 0.8126 | 0.8223 |                 2.67 |          0.5079 |
|  9 | bert-base-cased    |     100 |       10 | 0.8103 | 0.8198 |                 2.95 |          0.4578 |
| 11 | bert-base-uncased  |     512 |       05 | 0.8089 | 0.8183 |                16.77 |          0.0804 |
|  2 | bert-large-uncased |     512 |       05 | 0.8088 | 0.82   |                61.82 |          0.0218 |
|  7 | bert-base-uncased  |     300 |       20 | 0.8087 | 0.818  |                 4.85 |          0.2779 |
|  8 | bert-base-uncased  |     512 |       20 | 0.8053 | 0.8155 |                 9.18 |          0.1462 |

### Ordered after best f1 per second ratio, i.e. $\frac{f1 \times 100}{\text{time per epoch} (s)}$
|    | model              |   token |   epochs |     f1 |    acc |   time per epoch (m) |   f1 per second |
|---:|:-------------------|--------:|---------:|-------:|-------:|---------------------:|----------------:|
| 12 | bert-base-uncased  |     100 |       05 | 0.8126 | 0.8223 |                 2.67 |          0.5079 |
|  9 | bert-base-cased    |     100 |       10 | 0.8103 | 0.8198 |                 2.95 |          0.4578 |
| 10 | bert-base-uncased  |     100 |       20 | 0.8164 | 0.8257 |                 3.15 |          0.432  |
|  7 | bert-base-uncased  |     300 |       20 | 0.8087 | 0.818  |                 4.85 |          0.2779 |
|  1 | bert-large-uncased |     050 |       10 | 0.8178 | 0.828  |                 5.33 |          0.2556 |
|  8 | bert-base-uncased  |     512 |       20 | 0.8053 | 0.8155 |                 9.18 |          0.1462 |
| 13 | bert-base-uncased  |     300 |       05 | 0.8148 | 0.8238 |                10.08 |          0.1347 |
|  6 | bert-large-uncased |     100 |       10 | 0.8282 | 0.8364 |                10.83 |          0.1274 |
|  0 | bert-large-uncased |     100 |       20 | 0.828  | 0.8364 |                11.45 |          0.1205 |
| 11 | bert-base-uncased  |     512 |       05 | 0.8089 | 0.8183 |                16.77 |          0.0804 |
|  4 | bert-large-uncased |     200 |       10 | 0.8283 | 0.8364 |                20.35 |          0.0678 |
|  5 | bert-large-uncased |     300 |       05 | 0.823  | 0.8311 |                34.47 |          0.0398 |
|  3 | bert-large-uncased |     300 |       10 | 0.823  | 0.8325 |                38.9  |          0.0353 |
|  2 | bert-large-uncased |     512 |       05 | 0.8088 | 0.82   |                61.82 |          0.0218 |  




## New Evaluations results
p(i,x) means predicted x by model i. Otherwise the Categorie labels are the target labels    
Input is allways a String, either only "Text" or if specified the categorie labels concatenated with ". " before the text.


|N° | Model                 | Input                           |Output            |Cat1 accuracy| Cat2 accuracy|F1 score macro|
|:--|:----------------------|:--------------------------------|:-----------------|:-------------|:-------------|:-------------|
|0  | Classifier_multi_2Cat | Text                            | Cat1, Cat2       |0.8223        | 0.5404       |-             |
|2  | Classifier_lvl1       | Text                            | Cat1             |0.8242        | -            | -            |
|2.1| Classifier_lvl1 300 T 10 ep     | Text                  | Cat1             |0.8241        | -            | -            |
|2.2| Classifier_lvl1 100 T 10 ep     | Text                  | Cat1             |0.8234        | -            |0.8234        |
|2.3| Classifier_lvl1 512 T 10 ep     | Text                  | Cat1             |0.8230        | -            |0.8123        |
|3  | Classifier_lvl2_f     | Text                            | Cat2             |-             | 0.5879       | -            |
|4  | Classifier_lvl2_h     | Cat1. Text                      | Cat2             |-             | 0.6670       | -            |
|5  | Classifier_lvl2_h     | p(2, Cat1). Text                | Cat2             |-             | 0.6315       | -            |



## Evaluations results
Training the models for 5 epochs using a maximal token length of 100 and a batch size of 26.   
p(i,x) means predicted x by model i. Otherwise the Categorie labels are the target labels    
Input is allways a String, either only "Text" or if specified the categorie labels concatenated with ". " before the text.

|N°  | Model                 | Input                           |Output            |Cat1 accuracy| Cat2 accuracy| Cat3 accuracy|F1 score macro
|:--|:----------------------|:--------------------------------|:-----------------|:-------------|:-------------|:-------------|:-------------
|0  | Classifier_multi_2Cat | Text                            | Cat1, Cat2       |0.8223        | 0.5404       | -            |-
|1  | Classifier_multi_3Cat | Text                            | Cat1, Cat2, Cat3 |0.8203        | 0.5192       | 0            |-
|1.2| Classifier_multi_3Cat 10 epochs | Text                  | Cat1, Cat2, Cat3 |0.824         | 0.5532       | 0.0026       |-
|2  | Classifier_lvl1       | Text                            | Cat1             |0.8242        | -            | -            |-
|2.1| Classifier_lvl1 300 T 10 ep     | Text                  | Cat1             |0.8241        | -            | -            |-
|2.2| Classifier_lvl1 100 T 10 ep     | Text                  | Cat1             |0.8234        | -            | -            |0.8234
|2.3| Classifier_lvl1 512 T 10 ep     | Text                  | Cat1             |0.8230        | -            | -            |0.8123
|3  | Classifier_lvl2_f     | Text                            | Cat2             |-             | 0.5879       | -            |-
|4  | Classifier_lvl2_h     | Cat1. Text                      | Cat2             |-             | 0.6670       | -            |-
|5  | Classifier_lvl2_h     | p(2, Cat1). Text                | Cat2             |-             | 0.6315       | -            |-
|6  | Classifier_lvl3_f     | Text                            | Cat3             |-             | -            | 0.0041       |-
|7  | Classifier_lvl3_h     | Cat1. Cat2. Text                | Cat3             |-             | -            | 0.0204       |-
|8  | Classifier_lvl3_h     | p(2, Cat1). Cat2. Text          | Cat3             |-             | -            | 0.0196       |-
|9  | Classifier_lvl3_h     | Cat1. p(2, Cat4). Text          | Cat3             |-             | -            | 0.0076       |-
|10 | Classifier_lvl3_h     | p(2, Cat1). p(2, Cat4). Text    | Cat3             |-             | -            | 0.0076       |-


## Data token length distribution
### Amazon

#### Training data
![Data_distribution](./visualizations/amazon/Data_analysis.svg)
#### Test data
![Data_distribution](./visualizations/amazon/Data_analysis_test.svg)

### DBPedia

#### Training data
![Data_distribution](./visualizations/dbpedia/Data_analysis.svg)
#### Test data
![Data_distribution](./visualizations/dbpedia/Data_analysis_test.svg)


## Labels statistics
### Amazon
#### Training data
Amount of appearances for Cat1:
 * unique values 6  
 * Minimal: grocery gourmet food appears 3617 times  
 * Maximal: toys games appears 10266 times  
 * in average 6666.67 times.  


Amount of appearances for Cat2:
 * unique values 64  
 * Minimal: small animals appears 29 times  
 * Maximal: personal care appears 2852 times  
 * in average 625.00 times.  


Amount of appearances for Cat3:
 * unique values 464  
 * Minimal: aprons smocks appears 1 times  
 * Maximal: unknown appears 2262 times  
 * in average 86.21 times.  


#### Test data
Amount of appearances for Cat1:
 * unique values 6  
 * Minimal: baby products appears 698 times  
 * Maximal: health personal care appears 2992 times  
 * in average 1666.67 times.  


Amount of appearances for Cat2:
 * unique values 64  
 * Minimal: baby food appears 2 times  
 * Maximal: nutrition wellness appears 904 times  
 * in average 156.25 times.  


Amount of appearances for Cat3:
 * unique values 377  
 * Minimal: aquarium hoods appears 1 times  
 * Maximal: vitamins supplements appears 665 times  
 * in average 26.53 times.  

### DBPedia

#### Training data
Amount of appearances for Cat1: 
 * unique values 9  
 * Minimal: Device appears 248 times  
 * Maximal: Agent appears 124798 times  
 * in average 26771.33 times.  
 
Amount of appearances for Cat2: 
 * unique values 70  
 * Minimal: Database appears 129 times  
 * Maximal: Athlete appears 31111 times  
 * in average 3442.03 times.  
 
Amount of appearances for Cat3: 
 * unique values 219  
 * Minimal: BiologicalDatabase appears 129 times  
 * Maximal: AcademicJournal appears 1924 times  
 * in average 1100.19 times. 

#### Test Data
Amount of appearances for Cat1: 
 * unique values 9  
 * Minimal: Device appears 62 times  
 * Maximal: Agent appears 31495 times  
 * in average 6754.89 times.  
 
Amount of appearances for Cat2: 
 * unique values 70  
 * Minimal: Database appears 33 times  
 * Maximal: Athlete appears 7855 times  
 * in average 868.49 times.  
 
Amount of appearances for Cat3: 
 * unique values 219  
 * Minimal: BiologicalDatabase appears 33 times  
 * Maximal: AcademicJournal appears 485 times  
 * in average 277.60 times.
---

[Based on](https://towardsdatascience.com/multi-label-multi-class-text-classification-with-bert-transformer-and-keras-c6355eccb63a)  
[Pretrained models](https://huggingface.co/transformers/pretrained_models.html)  
[BERT docu](https://huggingface.co/transformers/model_doc/bert.html)  
[Hierarchical text classification](https://www.kaggle.com/kashnitsky/hierarchical-text-classification)
[DBPedia](https://www.kaggle.com/danofer/dbpedia-classes?select=DBPEDIA_train.csv)
