import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv(r"C:\Users\yildi\OneDrive\Masaüstü\datasets/amazon_review.csv")

df.head()

df.info()
df.shape
df["reviewerID"].nunique()
df["asin"].nunique()
df["overall"].nunique()
df["overall"].max()
df["overall"].min()
df["overall"].mean()
df.day_diff.value_counts()
df.day_diff.min()
df.day_diff.max()
df.day_diff.mean()

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 90, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 350), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 350) & (dataframe["day_diff"] <= 750), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 750), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

df["overall"].mean()

## güncel değerlerdirmeler ortalamayı yükseltmiştir. ##

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

def score_yes_no_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.

    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["score_yes_no_diff"] = df.apply(lambda x: score_yes_no_diff(x["helpful_yes"],
                                                                             x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                                 x["helpful_no"]), axis=1)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                             x["helpful_no"]), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)


