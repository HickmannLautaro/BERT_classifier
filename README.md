# BERT for hierarchicla text classification on a dataset with Amazon product reviews
Using the pretrained bert-base-uncased form [Hugging Face](https://huggingface.co/bert-base-uncased)   

Dataset with Amazon product reviews, classes are structured as: 
* 6 "level 1" classes 
* 64 "level 2" classes 
* 510 "level 3" classes

The training set contains $40k$ documents and the test set $10k$.  
Each document contains: Title, Text, Cat1, Cat2, Cat3  
## Model architectures
### All at once
![all_classses](./visualizations/3clases.svg)
### Hierarchical
![all_classses](./visualizations/hierarchical.svg)

## Evaluations results
Training the models for $5$ epochs using a maximal token length of $100$ and a batch size of $26$.   
$p(i,x)$ means predicted $x$ by model $i$. Otherwise the Categorie labels are the target labels    
Input is allways a String, either only "Text" or if specified the categorie labels concatenated with ". " before the text.

|N°  | Model                 | Input                           |Output            |Cat1 accuracy| Cat2 accuracy| Cat3 accuracy
|:---|:----------------------|:--------------------------------|:-----------------|:------------|:-------------|:------------
|$0$ | Classifier_multi_2Cat | Text                            | Cat1, Cat2       |$0.8223$     | $0.5404$     | $-$
|$1$ | Classifier_multi_3Cat | Text                            | Cat1, Cat2, Cat3 |$0.8203$     | $0.5192$     | $0$
|$2$ | Classifier_lvl1       | Text                            | Cat1             |$0.8242$     | $-$          | $-$
|$3$ | Classifier_lvl2_f     | Text                            | Cat2             |$-$          | $0.5879$     | $-$
|$4$ | Classifier_lvl2_h     | Cat1. Text                      | Cat2             |$-$          | $0.6670$     | $-$
|$5$ | Classifier_lvl2_h     | $p(2, Cat1)$. Text              | Cat2             |$-$          | $0.6315$     | $-$
|$6$ | Classifier_lvl3_f     | Text                            | Cat3             |$-$          | $-$          | $0.0041$
|$7$ | Classifier_lvl3_h     | Cat1. Cat2. Text                | Cat3             |$-$          | $-$          | $0.0204$
|$8$ | Classifier_lvl3_h     | $p(2, Cat1)$. Cat2. Text        | Cat3             |$-$          | $-$          | $0.0196$
|$9$ | Classifier_lvl3_h     | Cat1. $p(2, Cat4)$. Text        | Cat3             |$-$          | $-$          | $0.0076$
|$10$| Classifier_lvl3_h     | $p(2, Cat1)$. $p(2, Cat4)$. Text| Cat3             |$-$          | $-$          | $0.0076$
