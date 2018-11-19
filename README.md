# Word-Sense-Representation-and-Disambiguation
##評估 (兩種都要算)：
1. 分類器內建函式：nltk.classify.accuracy(sklearn_classifier, test_set) (未篩選)
2. 自行計算 accuracy：WordNet 分類共有2653個，但每個 group 只會對應到部分的分類，所以請同學根據 head word 所屬的 group，篩選分類器預測的結果，選擇預測機率最高者作為答案。

##舉例說明：
1. abandon v
> * Definition: to leave someone or something somewhere, sometimes not returning to get them
> * Example: They were forced to abandon the car.||All at once, they abandoned their sacks of stolen property and ran for their lives!
> * 篩選前結果: change.v.02 (不屬於 abandon-v 所對應到的 WordNet 分類)
> * 篩選後結果: leave.v.01<br>
{'abandon-v-1': 'get_rid_of.v.01', <br>
 'abandon-v-2': 'abandon.v.02', <br>
 'abandon-v-3': 'leave.v.01', <br>
 'abandon-v-4': 'abandon.v.04', <br>
 'abandon-v-5': 'leave.v.02'}
