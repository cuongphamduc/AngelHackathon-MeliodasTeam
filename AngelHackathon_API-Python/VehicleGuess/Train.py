from sklearn.ensemble import RandomForestRegressor
from VehicleGuess.FeatureManager import FeatureManager
from sklearn.externals import joblib

a = FeatureManager()
regr = RandomForestRegressor()
regr.fit(a.train_features, a.trainVehi)
joblib.dump(regr, 'result.pkl')