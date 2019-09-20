import numpy as np
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt


ads_file = 'data/ad_table.csv'
df = pd.read_csv(ads_file, header=0, sep=',')

df_agg = pd.DataFrame({'shown': df['shown'], 'clicked' : df['clicked'], 'converted' : df['converted'], \
                       'ad': df['ad']}).groupby('ad').sum()

ctr = pd.DataFrame({'shown_vs_clicked': df_agg['clicked']/df_agg['shown']})
cvr = pd.DataFrame({'clicked_vs_converted': df_agg['converted']/df_agg['clicked']})

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

join = pd.DataFrame({'shown_vs_clicked':ctr['shown_vs_clicked'],'clicked_vs_converted':\
                     cvr['clicked_vs_converted'],'profit':df.groupby('ad')['total_revenue'].mean()})
ad_data = np.array(join)

np.random.seed(5)
    
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
aa = KMeans(n_clusters=3).fit(ad_data)
labels = aa.labels_

ax.scatter(ad_data[:, 0], ad_data[:, 1], ad_data[:, 2],
            c=labels.astype(np.float), marker='o')

ax.set_xlabel('CTR')
ax.set_ylabel('CVR')
ax.set_zlabel('Profit')
ax.set_title('Ad Group Clustering based on 3 Performance Metrics')
ax.dist = 12
plt.show()