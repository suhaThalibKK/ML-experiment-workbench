"""Export actual verified Python models and results for browser exploration."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import shutil
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'output/computing_lab1'
RUN=ROOT/'output/computing_lab1_run'
DIST=Path(__file__).resolve().parents[1]/'dist'
ASSETS=DIST/'assets'
ASSETS.mkdir(exist_ok=True)

def load(number):
 p=next(BASE.glob(f'{number:02d}_*.py'))
 spec=importlib.util.spec_from_file_location('exp'+str(number),p)
 module=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=module
 spec.loader.exec_module(module)
 return module,p

def serial(obj):
 if isinstance(obj,(np.ndarray,pd.Index)): return obj.tolist()
 if isinstance(obj,np.generic): return obj.item()
 if isinstance(obj,Path): return str(obj)
 raise TypeError(type(obj))

def dataset(name):
 frame=pd.read_csv(BASE/'data'/f'{name}.csv',na_values=['?'])
 clean=frame.astype(object).where(frame.notna(),None)
 return {'columns':list(frame.columns),'rows':clean.values.tolist(),'count':len(frame)}

pack={'datasets':{name:dataset(name) for name in ['weather','iris','wine','heart','spam']},'experiments':[],'models':{}}
meta=[
 ('ID3 algorithm','Decision trees','Build a decision tree using information gain and classify new weather conditions.','Weather · 14 records','Categorical ID3','4 predictors · 2 classes',
  [('Training fit','14 / 14','Training data only'),('Root feature','Outlook','Highest information gain'),('New prediction','Yes','Sunny · normal humidity')]),
 ('Naive Bayes classifier','Probabilistic learning','Evaluate a Gaussian Naive Bayes classifier implemented from scratch on Iris and Wine.','Iris & Wine · 328 records','Gaussian Naive Bayes','80 / 20 stratified split',
  [('Iris accuracy','96.67%','29 of 30 test records'),('Wine accuracy','97.22%','35 of 36 test records'),('Implementation','From scratch','Stable log likelihoods')]),
 ('Document classifier','Natural language processing','Identify spam using a training-only word vocabulary and Multinomial Naive Bayes.','SMS · 5,574 documents','Multinomial Naive Bayes','4,127 train · 1,032 test',
  [('Accuracy','98.45%','1,016 of 1,032 messages'),('Spam precision','96.67%','116 true / 120 predicted'),('Spam recall','90.62%','116 of 128 spam messages')]),
 ('Heart disease diagnosis','Bayesian networks','Learn a discrete Bayesian network and infer a class from held-out Cleveland records.','Cleveland · 303 records','Bayesian network','242 train · 61 test',
  [('Test accuracy','86.89%','53 of 61 held-out records'),('Predictors','13','Training-only preprocessing'),('Network','14 nodes','Discrete CPDs · BDeu prior')]),
 ('EM & K-Means clustering','Unsupervised learning','Compare K-Means and an EM-fitted Gaussian mixture on the same standardized Iris features.','Iris · 150 records','K-Means & Gaussian mixture','3 clusters · 4 features',
  [('K-Means silhouette','0.4599','Higher silhouette in this run'),('EM silhouette','0.3742','Adjusted Rand index 0.9039'),('EM convergence','19 iterations','Full covariance · 3 components')]),
 ('K-nearest neighbors','Instance-based learning','Classify Iris with three nearest neighbors after fitting the scaler on training data.','Iris · 150 records','KNN · k = 3','90 train · 60 test',
  [('Test accuracy','95.00%','57 of 60 records'),('Correct predictions','57','3 incorrect predictions'),('Neighbors','k = 3','Euclidean distance · scaled')]),
 ('5-fold cross validation','Model evaluation','Compare accuracy, macro precision, recall and F1 across five stratified validation folds.','Iris · 150 records','Entropy decision tree','5 folds · 120 / 30 each',
  [('Mean accuracy','95.33%','Sample standard deviation 3.80%'),('Macro precision','0.9572','Equal weight per class'),('Macro F1','0.9531','Each row validated once')]),
 ('Google Colab familiarization','Cloud notebooks','Practice notebook execution, numerical operations, files, plotting and a simple ML workflow.','NumPy · pandas · Iris','Python notebook workflow','CPU runtime sufficient',
  [('CSV round trip','Passed','Written and read back correctly'),('Iris accuracy','96.67%','120 train · 30 test'),('Libraries','NumPy + pandas','Matrix operations & charts')]),
]
for number,record in enumerate(meta,1):
 module,path=load(number)
 title,category,description,data_label,method,split,metrics=record
 output=(RUN/(path.stem+'.txt')).read_text().replace(str(BASE/'results'),'./results')
 shutil.copy2(path,ASSETS/path.name)
 pack['experiments'].append({'id':number,'title':title,'category':category,'description':description,
  'datasetLabel':data_label,'method':method,'split':split,'metrics':[{'label':m[0],'value':m[1],'note':m[2]} for m in metrics],
  'source':path.read_text(),'filename':path.name,'output':output,
  'datasets':{1:['weather'],2:['iris','wine'],3:['spam'],4:['heart'],5:['iris'],6:['iris'],7:['iris'],8:['iris']}[number]})

id3,_=load(1)
weather=pd.read_csv(BASE/'data/weather.csv',dtype=str)
tree=id3.id3(weather,[c for c in weather if c not in {'id','play'}])
def tree_json(node):
 return {'prediction':str(node.prediction),'attribute':node.attribute,
         'children':{str(k):tree_json(v) for k,v in node.children.items()}}
pack['models']['id3']=tree_json(tree)
nb,_=load(2)
pack['models']['gaussian']={}
for name in ['iris','wine']:
 frame=pd.read_csv(BASE/'data'/f'{name}.csv')
 Xtr,Xte,ytr,yte=train_test_split(frame.iloc[:,:-1],frame.iloc[:,-1],test_size=.2,random_state=42,stratify=frame.iloc[:,-1])
 model=nb.GaussianNBFromScratch().fit(Xtr,ytr)
 preds=model.predict(Xte)
 pack['models']['gaussian'][name]={'classes':model.classes_,'means':model.means_,'variances':model.variances_,
  'logPriors':model.log_priors_,'features':list(frame.columns[:-1]),'defaults':Xte.iloc[0].values,
  'testRows':Xte.index.tolist(),'matrix':confusion_matrix(yte,preds,labels=model.classes_)}
doc,_=load(3)
with contextlib.redirect_stdout(io.StringIO()): pipeline,train_text,test_text=doc.main()
vec,classifier=pipeline.steps[0][1],pipeline.steps[1][1]
pack['models']['document']={'vocabulary':vec.vocabulary_,'classes':classifier.classes_,
 'featureLogProb':classifier.feature_log_prob_,'classLogPrior':classifier.class_log_prior_}
heartmod,_=load(4)
heartmod.RESULTS_DIR=ROOT/'tmp/computing_lab_review/site_model_exports'
with contextlib.redirect_stdout(io.StringIO()): model,test_d,states,probabilities=heartmod.main()
frame=pd.read_csv(BASE/'data/heart.csv',na_values=['?'])
train,test=train_test_split(frame,test_size=.2,random_state=42,stratify=frame.target)
_,_,_,rules=heartmod.discretize_training_and_test(train,test)
cpds=[{'variable':cpd.variable,'variables':cpd.variables,'cardinality':cpd.cardinality,
       'states':cpd.state_names,'values':cpd.values.flatten()} for cpd in model.get_cpds()]
pack['models']['heart']={'cpds':cpds,'rules':rules,'features':heartmod.FEATURES,'continuous':heartmod.CONTINUOUS,
 'states':states,'edges':list(model.edges()),'defaultRaw':test.iloc[0][heartmod.FEATURES].to_dict(),
 'defaultDiscrete':test_d.iloc[0].to_dict(),'firstPresence':probabilities[0],
 'predictions':pd.read_csv(RUN/'heart_predictions.csv').to_dict('records')}
iris=pd.read_csv(BASE/'data/iris.csv')
X=iris.iloc[:,:4].values
scaler=StandardScaler().fit(X)
scaled=scaler.transform(X)
gmm=GaussianMixture(n_components=3,covariance_type='full',n_init=10,max_iter=500,reg_covar=1e-6,random_state=42).fit(scaled)
km=KMeans(n_clusters=3,n_init=10,random_state=42).fit(scaled)
assign=pd.read_csv(RUN/'cluster_assignments.csv')
assert np.array_equal(km.labels_,assign.kmeans)
assert np.array_equal(gmm.predict(scaled),assign.em_gmm)
pack['models']['clustering']={'scaled':scaled,'kmeans':assign.kmeans.values,'em':assign.em_gmm.values,
 'kmeansCenters':km.cluster_centers_,'emMeans':gmm.means_,'memberships':gmm.predict_proba(scaled)}
Xtr,Xte,ytr,yte=train_test_split(iris.iloc[:,:4],iris.species,test_size=.4,random_state=42,stratify=iris.species)
scaler=StandardScaler().fit(Xtr)
pack['models']['knn']={'trainX':scaler.transform(Xtr),'trainY':ytr.to_numpy(),'mean':scaler.mean_,'scale':scaler.scale_,
 'trainRows':Xtr.index,'default':Xte.iloc[0].values,'predictions':pd.read_csv(RUN/'knn_predictions.csv').to_dict('records')}
cv=StratifiedKFold(5,shuffle=True,random_state=42)
folds=[]
for i,(train_idx,test_idx) in enumerate(cv.split(X,iris.species),1):
 tree=DecisionTreeClassifier(criterion='entropy',random_state=42).fit(X[train_idx],iris.species.iloc[train_idx])
 pred=tree.predict(X[test_idx]); true=iris.species.iloc[test_idx]
 folds.append({'fold':i,'accuracy':accuracy_score(true,pred),'precision':precision_score(true,pred,average='macro',zero_division=0),
  'recall':recall_score(true,pred,average='macro',zero_division=0),'f1':f1_score(true,pred,average='macro',zero_division=0),
  'validationRows':test_idx,'matrix':confusion_matrix(true,pred,labels=tree.classes_)})
pack['models']['cv']={'folds':folds,'classes':['setosa','versicolor','virginica']}
(ASSETS/'lab-data.json').write_text(json.dumps(pack,default=serial,ensure_ascii=True,allow_nan=False,separators=(',',':')))
for path in (BASE/'data').iterdir():
 if path.suffix=='.csv': shutil.copy2(path,ASSETS/path.name)
for name in ['heart_network.png','clustering_comparison.png','colab_demo.png']:
 shutil.copy2(RUN/name,ASSETS/name)
shutil.copy2(ROOT/'output/pdf/Computing_Lab1_Consolidated_Record.pdf',ASSETS/'Computing_Lab1_Record.pdf')
shutil.copy2(BASE/'Computing_Lab1.ipynb',ASSETS/'Computing_Lab1.ipynb')
shutil.copy2(ROOT/'output/Computing_Lab1_Corrected.zip',ASSETS/'Computing_Lab1_Programs.zip')
shutil.copy2(BASE/'data/SOURCES.md',ASSETS/'DATA_SOURCES.md')
print('Exported 8 experiments, 5 full datasets, fitted models, verified outputs and downloads.')
print('Data JSON:',(ASSETS/'lab-data.json').stat().st_size,'bytes')
