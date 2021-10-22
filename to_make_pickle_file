#K-meas法のpickleファイルを作成するためのコードです

import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans

#Numpy配列に変換する
def img_to_matrix(img):
  img_array = np.asarray(img)
  return img_array

#配列を平坦化する
def flatten_img(img_array):
  s = img_array.shape[0] * img_array.shape[1] * img_array.shape[2]
  img_width = img_array.reshape(1, s)
  #print(img_width)
  return img_width[0]

dataset = []
img_paths = []

for file in glob.glob("./drive/MyDrive/rarelity/new_rarelity/rarelity/*.PNG"):
  img_paths.append(file)

print("Image number:", len(img_paths))
print("Image list make done.")
#print(img_paths)

for i in img_paths:
  img = Image.open(i)
  img = img_to_matrix(img)
  img = flatten_img(img)
  dataset.append(img)

print(type(dataset))
dataset = np.array(dataset)
print(dataset)
print(dataset.shape)
print("Dataset make done.")


# K-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=5).fit(dataset)
labels = kmeans.labels_
print("K-means clustering done.")

for i in range(n_clusters):
  label = np.where(labels==i)[0]
  # Image placing
  if not os.path.exists("label"+str(i)):
    os.makedirs("label"+str(i))

  for j in label:
    img = Image.open(img_paths[j])
    fname = img_paths[j].split('/')[-1]
    img.save("label"+str(i)+"/" + fname)

print("Image placing done.")

#pickleファイルとして保存
import pickle
with open('uma_kmeans_test2.pkl', 'wb') as fp:
  pickle.dump(kmeans, fp)


#次元削減(グラフ化用)
#n = dataset.shape[0]
#print(n//180)
#batch_size = 180
#ipca = IncrementalPCA(n_components=2)
#print(ipca)
#for i in range(n//batch_size):
#  r_dataset = ipca.partial_fit(dataset[i*batch_size:(i+1)*batch_size])
#  r_dataset = ipca.transform(dataset)
#  print(r_dataset.shape)
#print(r_dataset)
#print(r_dataset.shape)
#print("PCA done.")

#クラスタリングのグラフ化（wipは分類後の結果を見て並び替え必要）
#wip = [("SSR","magenta"), ("R","gray"), ("SR","y")] 

#for i in range(3):
#  data = []
#  for j in range(len(labels)):
#    if labels[j] == i:
#      data.append(list(r_dataset[j]))
#  data = np.array(data)
#  #print(data)
#  print(type(data))
#  plt.scatter(data[:, 0], data[:, 1], facecolors='none', edgecolors= wip[i][1], label = wip[i][0])
#  plt.legend(loc="best")
#  plt.axis("on")
#  #plt.yticks(color="None")
#  #plt.xticks(color="None")
#  plt.savefig("img3.png")
