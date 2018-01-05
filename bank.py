
# coding: utf-8

# ## Data Input

# In[3]:

get_ipython().magic('matplotlib inline')
import pandas as pd
pd.options.display.mpl_style = 'default'

data_path = u'/user/BANK_DATA.csv'
rawData = sc.textFile(data_path)

print('Number of raw data: %d'%rawData.count())
print('Samples of raw data:')
pd.DataFrame(rawData.take(5))


# ## Data Pre-processing

# In[17]:

#delete comma
no_comma_data = rawData.map(lambda x: x.split(',')).map(lambda x: [xx.replace('\'','') for xx in x])  

#delete unknown data
def deleteUnknownForLine(data):
    flag = True
    if 'unknown' in data:
        flag = False
    return flag

data = no_comma_data.filter(lambda x: deleteUnknownForLine(x))
data.cache()
print('Preprocess input data done!')
print('Samples of pre-processed data:')
pd.DataFrame(data.take(5))


# ## Data Visualization

# In[18]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
get_ipython().magic('matplotlib inline')

#Statistical age distribution
print('Age distribution:')  
ages = data.map(lambda x: float(x[0])).collect()
hist(ages, bins = 20, color='lightblue', normed = False)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
    
#Statistical oppupational distribution
num_occupations = data.map(lambda x: x[1]).distinct().count()
print("Oppupational number: %d"%(num_occupations))
print('Oppupational distribution：')

count_by_occupation = data.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y).collect()

x_axis1 = np.array([c[0] for c in count_by_occupation])
y_axis1 = np.array([c[1] for c in count_by_occupation])
x_axis = x_axis1[np.argsort(y_axis1)]
y_axis = y_axis1[np.argsort(y_axis1)]
pos = np.arange(len(x_axis))
width = 1.0
ax = plt.axes()
ax.set_xticks(pos+(width)/2)
ax.set_xticklabels(x_axis)
    
plt.bar(pos, y_axis, width, color = 'lightblue')
plt.xticks(rotation=30)
plt.title('Occupation Distribution')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.show()

print('Visualization Done！')


# ## Modeling

# In[12]:

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest

nominal_index = [1, 2, 3, 4, 5]#column index of category feature
def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()
mappings = [get_mapping(data, i) for i in nominal_index]
print "Mappings:"
print mappings

def extract_label(data):
    label = 1 if data[-1] == 'yes' else 0
    return label

def extract_features(data, mappings, nominal_index): 
    cat_vec = np.zeros(len(nominal_index)) 
    for index_map, index_data in enumerate(nominal_index):
        dict_code = mappings[index_map]
        cat_vec[index_map] = dict_code[data[index_data]]
    feature = np.concatenate(([float(data[0])], cat_vec))
    return feature

label_feature = data.map(lambda point: LabeledPoint(extract_label(point), extract_features(point, mappings, nominal_index))) 

(trainData, testData) = label_feature.randomSplit([0.7, 0.3])##split data into training data and test data
model = RandomForest.trainClassifier(trainData, numClasses = 2, categoricalFeaturesInfo={}, numTrees = 10, featureSubsetStrategy="auto", impurity = 'gini', maxDepth = 10, maxBins = 50)

print('\nTrain model done.')


# ## Model Application

# In[13]:

from pyspark.mllib.evaluation import BinaryClassificationMetrics

scores = model.predict(testData.map(lambda x: x.features))
all_models_metrics = []
scoresAndLabels = testData.map(lambda x: x.label).zip(scores)
metrics = BinaryClassificationMetrics(scoresAndLabels)  #evalution
all_models_metrics.append((model.__class__.__name__, metrics.areaUnderROC, metrics.areaUnderPR))

for modelName, AUC, PR in all_models_metrics:
    print ('AUC is %f, PR is %f'%(AUC, PR))
print('Evalution model done.')


# ## Model Prediction

# In[33]:

result = model.predict(testData.map(lambda x: x.features))
print(result.take(30))
df = spark.createDataFrame(rdd, ['name', 'age'])


# ## Model Output

# In[32]:

rdd = sc.parallelize(l)


# In[19]:

model_save_path = u'/user/RFModel'  
model.save(sc, model_save_path)


# ## Model Input

# In[8]:

from pyspark.mllib.tree import RandomForestModel

model_path = u'/user/RFModel'
new_model = RandomForestModel.load(sc, model_path)#load model
print('Load model done.')

