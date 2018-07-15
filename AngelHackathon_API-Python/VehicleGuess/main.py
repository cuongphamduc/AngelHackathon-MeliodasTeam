from sklearn.externals import joblib
import datetime
from sklearn.preprocessing import OneHotEncoder
from VehicleGuess.FeatureManager import FeatureManager

a = FeatureManager()

print("Nhập giờ : ")
h = (int)(input())
print("Nhập phút : ")
p = (int)(input())
print("Nhập ngày : ")
d = (int)(input())
print("Nhập tháng : ")
m = (int)(input())
print("Nhập năm : ")
y = (int)(input())
dt = datetime.datetime(y, m, d, h, p, 0)
time = []
time.append([dt.year, dt.month, dt.day, dt.weekday(), dt.hour, dt.minute])

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit(a.trainTime)
b = onehot_encoder.transform(time)

feature = []
tmp = []
tmp.extend(b[0])
c = [0, 0, 0]
tmp.extend(c)
feature.extend([tmp])

regr = joblib.load('result.pkl')
res = regr.predict(feature)
print("Lưu lượng xe dự đoán : ", res[0], "%")