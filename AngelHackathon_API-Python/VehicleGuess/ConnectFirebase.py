import pyrebase
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from VehicleGuess.FeatureManager import FeatureManager

a = FeatureManager()

config = {
  "apiKey": "AIzaSyD63ncTGMW430oLF-ScwE8TIdKEn15v3J8",
  "authDomain": "traffic-volume-map.firebaseapp.com",
  "databaseURL": "https://traffic-volume-map.firebaseio.com",
  "storageBucket": "traffic-volume-map.appspot.com"
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()
camera = db.child("camera").get()
print(camera.val())

def getVolumeGuess(guessTime):
    dt = datetime.datetime.strptime(guessTime, "%Y%m%d%H%M")
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
    return res[0]


def writeDataToFirebase(guessVolume):
    db.child("camera").child(0).update({"guess": guessVolume})


def stream_handler(message):
    guessTime = message["data"]
    print(guessTime)
    guessVolume = getVolumeGuess(guessTime)
    writeDataToFirebase(guessVolume)

my_stream = db.child("guess").child("guess-time").stream(stream_handler)