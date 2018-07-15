import datetime
from sklearn.preprocessing import OneHotEncoder
import pymysql

class FeatureManager:
    def __init__(self):
        db = pymysql.connect(user='root', password='Cuong@97', host='localhost', database='student')
        cursor = db.cursor()
        a = 0
        b = 0
        c = 0
        self.trainTime = []
        self.trainVehi = []
        self.trainVehiPrev = []
        sql = "SELECT * FROM TRAINDATA WHERE TIME < '%s'" % ('2018-07-03 00:00:00')
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results:
                dt = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                self.trainTime.append([dt.year, dt.month, dt.day, dt.weekday(), dt.hour, dt.minute])
                self.trainVehiPrev.append([a, b, c])
                a = row[1]
                b = a
                c = b
                self.trainVehi.append(row[1])
        except:
            print("Error: unable to fetch data")

        self.srcTime = []
        self.srcVehi = []
        self.testTime = []
        self.testVehi = []
        self.testVehiPrev = []
        sql = "SELECT * FROM TRAINDATA WHERE TIME >= '%s'" % ('2018-07-03 00:00:00')
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results:
                self.srcTime.append(row[0])
                self.srcVehi.append(row[1])
                dt = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                self.testTime.append([dt.year, dt.month, dt.day, dt.weekday(), dt.hour, dt.minute])
                self.testVehiPrev.append([a, b, c])
                a = row[1]
                b = a
                c = b
                self.testVehi.append(row[1])
        except:
            print("Error: unable to fetch data")
        db.close()

        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(self.trainTime)
        train_features = onehot_encoder.transform(self.trainTime)
        test_features = onehot_encoder.transform(self.testTime)

        self.train_features = []
        self.test_features = []
        for i in range(len(train_features)):
            tmp = []
            tmp.extend(train_features[i])
            tmp.extend(self.trainVehiPrev[i])
            self.train_features.extend([tmp])
        for i in range(len(test_features)):
            tmp = []
            tmp.extend(test_features[i])
            tmp.extend(self.testVehiPrev[i])
            self.test_features.extend([tmp])