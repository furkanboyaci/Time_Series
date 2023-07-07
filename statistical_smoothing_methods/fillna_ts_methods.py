##################################################
# Smoothing Methods (Holt-Winters)
##################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

warnings.filterwarnings('ignore')


############################
# Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

#Amacımız bir ay sonraki periyotta hava kirliliği seviyesinin tahmin edilmesi

data = sm.datasets.co2.load_pandas()
y = data.data

y.head() #veri seti haftalık

#Veri setinin frekansını aylığa çevirelim. Aynı ay içerisindeki gözlemlerin ortalamasını alalım.
y = y['co2'].resample('MS').mean() #MS : aylık frekansı belirtir.

#Dikkat: Zaman serilerinde eksik değerleri ortalama,medyan ile doldurmak değil de kendisinden önceki ya da sonraki değerlerle doldurulabilir. Ya da kendisinden önceki ve sonraki değerlerin ortalamasını almak şeklinde doldurulabilir. Tüm veri setinin ortalama-medyan değerini kullanmak sağlıksızdıe. Çünkü seride herhangi bir trend ya da mevsimsellik olduğu durumunda ilgili zaman periyotlarının değerleri biribirinden epey farklı olacaktır.

y.isnull().sum()

na_values = y[y.isnull()] #eksik gözlemler

# Geri Dolum (Backward Fill):

y_backward = na_values.fillna(y.bfill())

############################################
# İleri Dolum (Forward Fill):
y_forward = na_values.fillna(y.ffill())

############################################
# Lineer İnterpolasyon
y_linear_inter = y.interpolate(method='linear')

y_linear_inter = y_linear_inter[na_values.index]

############################################
# Spline İnterpolasyon
from scipy.interpolate import splrep, splev
import numpy as np

# Eksik değerlerin indekslerini alın
missing_indices = y.isnull()

# Eksik olmayan değerlerin indekslerini alın
valid_indices = ~missing_indices

# Eksik olmayan değerlerin zamanlarını ve değerlerini alın
x = np.arange(len(y))[valid_indices].astype(np.float64)
y_valid = y[valid_indices]

# Spline eğrisini oluşturun
spline = splrep(x, y_valid)

# Eksik değerlerin indekslerini alın
missing_indices = missing_indices[missing_indices].index

# Eksik değerlerin indekslerini 0'dan başlayacak şekilde indeksleyin
missing_indices_shifted = np.arange(len(missing_indices))

# Eksik değerlerin zamanlarını float64 veri tipine dönüştürün ve indeks kayması yapın
x_missing = missing_indices_shifted.astype(np.float64)

# Eksik değerlerin değerlerini spline ile tahmin edin
y_missing = splev(x_missing, spline)

# Eksik değerleri doldurun
y.loc[missing_indices] = y_missing

y.isnull().sum()

y_spline_inter = y[missing_indices]

###########################################

# Doldurma İşlemlerini Hep Birlikte İnceleyelim
y_backward
y_forward
y_linear_inter
y_spline_inter


all_fillna = pd.concat([y_backward, y_forward, y_linear_inter, y_spline_inter], axis=1)

###################################


























