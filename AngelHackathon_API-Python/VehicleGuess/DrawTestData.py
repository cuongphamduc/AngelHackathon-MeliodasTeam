from VehicleGuess.FeatureManager import FeatureManager
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

a = FeatureManager()

regr = joblib.load('result.pkl')
dstVehi = regr.predict(a.test_features)

tick_x = range(len(a.srcTime))[::500]
tick_time = a.srcTime[::500]
plt.plot(a.srcTime, a.srcVehi, label = "Dữ liệu thực tế")
plt.plot(a.srcTime, dstVehi, "r", label = "Dữ liệu dự đoán")
plt.xlabel('Time')
plt.ylabel('Vehicle')
plt.legend(loc='upper left')
plt.xticks(tick_x, tick_time, rotation=45, ha="right")
plt.ylim(0, np.max(a.srcVehi) * 2)
plt.title("MSE = " + (str)(mean_squared_error(dstVehi, a.srcVehi)))
plt.show()