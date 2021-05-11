#!/usr/bin/env python
# coding: utf-8

# # Ctlist

# In[ ]:


import pandas as pd


# In[ ]:


comment=pd.read_csv('/RawData.csv')


# In[ ]:


comment


# In[ ]:


new_comment=comment.drop(columns=['Listing URL','Posted Date'])
new_comment.dropna(inplace=True)
new_comment['total']=new_comment['Title'].str.cat(new_comment['Post Contents'])


# In[ ]:


new_comment.drop(columns=['Title','Post Contents'],inplace=True)


# In[ ]:


new_comment['total']


# In[ ]:


import string
#make translator object
translator=str.maketrans('','',string.punctuation)
# new_comment['total']=new_comment['total'].translate(translator)
new_comment['total']=new_comment['total'].apply(lambda x: x.translate(translator))


# In[ ]:


new_comment


# In[ ]:


from nltk.tokenize import word_tokenize
new_comment['total']=new_comment['total'].apply(word_tokenize)


# In[ ]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
new_comment['total'] = new_comment['total'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.


# In[ ]:


pd.set_option('max_colwidth',10000)


# In[ ]:


new_comment


# In[ ]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)


# In[ ]:


from nltk.corpus import stopwords

words = stopwords.words('english')
for w in ['!',',','.','?','-s','-ly','</s>','s']:
    stop_words.append(w)
new_comment['total'] = [w for w in new_comment['total'] if w not in stop_words]


# In[ ]:


new_comment


# In[ ]:


new_comment['total'] = new_comment['total'].apply(lambda x: " ".join(x))


# In[ ]:


new_comment


# In[ ]:


total_list=[]
for i in new_comment['total']:
    total_list.append(i)


# In[ ]:


len(total_list)


# In[ ]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(total_list)]


# In[ ]:


model = Doc2Vec(documents, vector_size=20, window=2, min_count=1, workers=4)


# In[ ]:





# In[ ]:




# In[ ]:


doc_v_list=[]
len(total_list)
for i in range(len(total_list)):
    doc_v_list.append(model.docvecs[i])


# In[ ]:


doc_v_list


# In[ ]:


import numpy as np
doc_v_list=np.array(doc_v_list)


# In[ ]:


import faiss
index=faiss.IndexFlatL2(20)
print(index.is_trained)
index.add(doc_v_list)                  # add vectors to the index
print(index.ntotal)


# In[ ]:


k=4
D,I=index.search(doc_v_list[:],k)
print(I)
print(D)


# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(doc_v_list)
km_labels=kmeans.labels_
kmeans.labels_


# In[ ]:


df=pd.DataFrame(doc_v_list)
df['labels'] = kmeans.labels_


# In[ ]:


df1 = df[df['labels']==0]
df2 = df[df['labels']==1]
df3 = df[df['labels']==2]


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(9,6)) 
plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',
    df3[0],df3[1],'g*')
plt.show()


# # FaceBook

# In[ ]:


Fcomment=pd.read_csv('RawData FB.csv')


# In[ ]:


Fcomment=Fcomment.drop(columns='Name')


# In[ ]:


Fcomment


# In[ ]:


import string
#make translator object
translator=str.maketrans('','',string.punctuation)
# new_comment['total']=new_comment['total'].translate(translator)
Fcomment['Post']=Fcomment['Post'].apply(lambda x: x.translate(translator))


# In[ ]:


from nltk.tokenize import word_tokenize
Fcomment['Post']=Fcomment['Post'].apply(word_tokenize)


# In[ ]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
Fcomment['Post'] = Fcomment['Post'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.


# In[ ]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)


# In[ ]:


from nltk.corpus import stopwords

words = stopwords.words('english')
for w in ['!',',','.','?','-s','-ly','</s>','s']:
    stop_words.append(w)
Fcomment['Post'] = [w for w in Fcomment['Post'] if w not in stop_words]


# In[ ]:


import logging
#打印日志
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)


# In[ ]:


Fcomment['Post'] = Fcomment['Post'].apply(lambda x: " ".join(x))


# In[ ]:


Fcomment


# In[ ]:


total_listF=[]
for i in Fcomment['Post']:
    total_listF.append(i)


# In[ ]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documentsF = [TaggedDocument(doc, [i]) for i, doc in enumerate(total_listF)]


# In[ ]:


modelF = Doc2Vec(documentsF, vector_size=20, window=2, min_count=1, workers=4)


# In[ ]:


doc_v_listF=[]
len(total_list)
for i in range(len(total_listF)):
    doc_v_listF.append(model.docvecs[i])


# In[ ]:


doc_v_listF


# In[ ]:


import numpy as np
doc_v_listF=np.array(doc_v_listF)


# In[ ]:


import faiss
indexF=faiss.IndexFlatL2(20)
print(indexF.is_trained)
indexF.add(doc_v_listF)                  # add vectors to the index
print(indexF.ntotal)


# In[ ]:


k=4
D,I=indexF.search(doc_v_listF[:],k)
print(I)
print(D)


# In[ ]:





# In[ ]:


from sklearn.cluster import KMeans
kmeansF = KMeans(n_clusters=3, random_state=0).fit(doc_v_listF)
km_labelsF=kmeansF.labels_
kmeansF.labels_


# In[ ]:


dfF=pd.DataFrame(doc_v_listF)
dfF['labels'] = kmeansF.labels_
df1F = dfF[dfF['labels']==0]
df2F = dfF[dfF['labels']==1]
df3F = dfF[dfF['labels']==2]
import matplotlib.pyplot as plt
figF = plt.figure(figsize=(9,6)) 
plt.plot(df1F[0],df1F[1],'bo',df2F[0],df2F[1],'r*',
    df3F[0],df3F[1],'g*')
plt.show()


# # Twitter

# In[ ]:


Tcomment=pd.read_csv('Twitter Comment.csv')


# In[ ]:


Tcomment


# In[ ]:


import string
#make translator object
translator=str.maketrans('','',string.punctuation)
# new_comment['total']=new_comment['total'].translate(translator)
Tcomment['Comment']=Tcomment['Comment'].apply(lambda x: x.translate(translator))


# In[ ]:


from nltk.tokenize import word_tokenize
Tcomment['Comment']=Tcomment['Comment'].apply(word_tokenize)


# In[ ]:


Tcomment


# In[ ]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
Tcomment['Comment'] = Tcomment['Comment'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.


# In[ ]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)
from nltk.corpus import stopwords
words = stopwords.words('english')
for w in ['!',',','.','?','-s','-ly','</s>','s']:
    stop_words.append(w)
Tcomment['Comment'] = [w for w in Tcomment['Comment'] if w not in stop_words]


# In[ ]:


import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)


# In[ ]:


Tcomment['Comment'] = Tcomment['Comment'].apply(lambda x: " ".join(x))


# In[ ]:


total_listT=[]
for i in Tcomment['Comment']:
    total_listT.append(i)


# In[ ]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documentsT = [TaggedDocument(doc, [i]) for i, doc in enumerate(total_listT)]


# In[ ]:


modelT = Doc2Vec(documentsT, vector_size=20, window=2, min_count=1, workers=4)


# In[ ]:


doc_v_listT=[]
for i in range(len(total_listT)):
    doc_v_listT.append(model.docvecs[i])


# In[ ]:


import numpy as np
doc_v_listT=np.array(doc_v_listT)


# In[ ]:


import faiss
indexT=faiss.IndexFlatL2(20)
print(indexT.is_trained)
indexT.add(doc_v_listT)                  # add vectors to the index
print(indexT.ntotal)


# In[ ]:


k=4
D,I=indexT.search(doc_v_listT[:],k)
print(I)
print(D)


# In[ ]:


from sklearn.cluster import KMeans
kmeansT = KMeans(n_clusters=3, random_state=0).fit(doc_v_listT)
km_labelsT=kmeansT.labels_
kmeansT.labels_


# In[ ]:


dfT=pd.DataFrame(doc_v_listT)
dfT['labels'] = kmeansT.labels_
df1T = dfT[dfT['labels']==0]
df2T = dfT[dfT['labels']==1]
df3T = dfT[dfT['labels']==2]
import matplotlib.pyplot as plt
figT = plt.figure(figsize=(9,6)) 
plt.plot(df1T[0],df1T[1],'bo',df2T[0],df2T[1],'r*',
    df3T[0],df3T[1],'g*')
plt.show()


# In[ ]:


#example
figTe = plt.figure(figsize=(9,6)) 
plt.plot(df3T[0],df3T[1],'g*')
plt.show()


# In[ ]:





