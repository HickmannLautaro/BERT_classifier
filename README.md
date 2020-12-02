# BERT for hierarchicla text classification on a dataset with Amazon product reviews
Using the pretrained bert-base-uncased form [Hugging Face](https://huggingface.co/bert-base-uncased)   

Dataset with Amazon product reviews, classes are structured as:
* 6 "level 1" classes
* 64 "level 2" classes
* 510 "level 3" classes

The training set contains 40k documents and the test set 10k.  
Each document contains: Title, Text, Cat1, Cat2, Cat3  
## Model architectures

### All at once
![all_classses](./visualizations/3clases.svg)

### Hierarchical
![hierarchical](./visualizations/hierarchical.svg)

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
### Training data
![Data_distribution](./visualizations/Data_analysis.svg)
### Test data
![Data_distribution](./visualizations/Data_analysis_test.svg)


## Labels statistics
### Training data
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


### Test data
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

---

[Based on](https://towardsdatascience.com/multi-label-multi-class-text-classification-with-bert-transformer-and-keras-c6355eccb63a)  
[Pretrained models](https://huggingface.co/transformers/pretrained_models.html)  
[BERT docu](https://huggingface.co/transformers/model_doc/bert.html)  
[Hierarchical text classification](https://www.kaggle.com/kashnitsky/hierarchical-text-classification)
