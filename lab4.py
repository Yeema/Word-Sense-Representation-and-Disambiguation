import re
from collections import defaultdict, Counter
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
import nltk, random
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier 
from sklearn.linear_model import LogisticRegression

def words(text): return re.findall(r'\w+', text.lower())
def wnTag(pos): return {'noun': 'n', 'verb': 'v', 'adjective': 'a', 'adverb': 'r'}[pos]

TF = defaultdict(lambda: defaultdict(lambda: 0))
DF = defaultdict(lambda: [])
training=[]
def trainLesk():
    training = [  line.strip().split('\t') for line in open('wn.in.evp.cat.txt', 'r') if line.strip() != '' ]

    def isHead(head, word, tag):
        try:
            return lmtzr.lemmatize(word, tag) == head
        except:
            return False
            
    # zucchini-n-2	vegetable.n.01	zucchini courgette||small cucumber-shaped vegetable marrow; typically dark green||	
    # {'zucchini-n-1': 'vine.n.01', 'zucchini-n-2': 'vegetable.n.01'}        
    for wnid, wncat, senseDef, target in training:
        head, pos = wnid.split('-')[:2]
        for word in words(senseDef):
            if word != head and not isHead(head, word, pos):
                TF[word][wncat] += 1
                DF[word] += [] if wncat in DF[word] else [wncat] 

trainLesk()

def gender_features(tn):
    tokens=nltk.word_tokenize(tn[2])
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for w in tokens:
        if w.lower() not in stopWords:
            wordsFiltered.append(w)
    word_cnt=Counter()
    print(wordsFiltered)
    for t in set(wordsFiltered):
        if t.isalpha() and len(DF[t.lower()])>0:
            cnt=Counter(TF[t])
            word_cnt[t]=sum(cnt.values())/len(DF[t.lower()])
    head=tn[0].split('-')
    head=head[0]+'-'+head[1]
    nums=word_cnt.most_common()[::-1]
    pos=0
    for i in range(len(nums)):
        if nums[i]!=0:
            pos=i
            break
    nums=nums[pos:pos+5]
    features={a.lower(): b for a,b in nums}
    features.update({'baseword':head})
    return features

def LG_gender(train_set, ts):
    print('== SkLearn MaxEnt ==')
    sklearn_classifier = SklearnClassifier(LogisticRegression(C=10e5)).train(train_set)
    return sklearn_classifier

featuresets = [(gender_features(tn), tn[1]) for tn in training]
random.shuffle(featuresets)
split_point = len(featuresets)*9//10
train_set, test_set = featuresets[:split_point], featuresets[split_point:]
LG_classifier=LG_gender(train_set, test_set)
hits=0
for ts in test_set:
    possibles = LG_classifier.prob_classify(ts[0])._prob_dict
    for train in training:
        if ts[0]['baseword'] in train[0]:
            checking=[val for key,val in eval(train[3]).items()]
            break
#     print(checking)
    temp = {}
    for wncat in checking:
        temp[wncat] = possibles[wncat] if wncat in possibles else 0
    checking = sorted(checking, key=lambda x: temp[x], reverse=True)
#     print(checking)
    if ts[1] == checking[0]:
        hits += 1

print(hits)
print(hits/len(test_set))