import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Read in the data
#Data = pd.read_csv('Full_Data.csv', encoding = "ISO-8859-1")
#Data.head(1)
data = pd.read_csv('mycsv.csv', encoding = "ISO-8859-1")

train =data.iloc[0:3300,0:]
test =data.iloc[100:,0:]

# Removing punctuations
slicedData= train.iloc[:,2:27]
slicedData.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
slicedData.columns= new_Index
slicedData.head(5)

# Convertng headlines to lower case
for index in new_Index:
    slicedData[index]=slicedData[index].str.lower()
slicedData.head(1)

headlines = []
for row in range(0,len(slicedData.index)):
    headlines.append(' '.join(str(x) for x in slicedData.iloc[row,0:25]))    
    headlines[0]
    
basicvectorizer = CountVectorizer(ngram_range=(1,1))
basictrain = basicvectorizer.fit_transform(headlines)

basicmodel = RandomForestClassifier(n_estimators=200, criterion='entropy',max_features='auto')
basicmodel = basicmodel.fit(basictrain, train["Label"])

testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
basictest = basicvectorizer.transform(testheadlines)
predictions = basicmodel.predict(basictest)

predictions

#pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])

print(basictrain.shape)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score 

print (classification_report(test["Label"], predictions))
print (accuracy_score(test["Label"], predictions))


# Saving model to disk
pickle.dump(basicmodel, open('model.pkl','wb'))
pickle.dump(basicvectorizer, open('modelvect.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict(basictest[377]))
print(model.predict(basictest[-1]))