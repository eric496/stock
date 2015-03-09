import pandas as pd
import numpy as np
import numpy as np
import KNNLearner
from matplotlib import pyplot as plt

#look back 100 days
lookback_days = 100
#forecast days
lookforward_days = 5
#number of files
file_count = 200
#number of rows skipped to read in the file
skip_rows = 1684
#number of rows need to read
num_rows = 1261
#initialize training dataset with five features
Xtrain = np.zeros(((num_rows - lookforward_days - lookback_days) * file_count, 5))
Ytrain = np.zeros(((num_rows - lookforward_days - lookback_days) * file_count, 1))
#normalize data
def normalizer(fname, skip_rows, num_rows, lookback_days):
    #get the data - plus 5 days at the end
    df = pd.io.parsers.read_csv(fname, skiprows = skip_rows, names = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'], nrows = num_rows, delimiter = ',')
    #get the adjusted close price
    data = np.array(df[['adj_close']])
    #reverse the array
    data = data[::-1]
    #historical price
    #h_data = data[:len(data) - 5].copy()
    #normalized price
    for k in range(len(data) - 5):
        data[k] = data[k + 5] / data[k] - 1
    #remove the last five rows 
    data = data[:len(data) - 5]
    train_y = data[lookback_days:]
    return data, train_y

for i in range(file_count):
    if i in range(10):
        fname = 'ML4T-00' + str(i) + '.csv'
    elif i in range(10, 100):
        fname = 'ML4T-0' + str(i) + '.csv'
    else:
        fname = 'ML4T-' + str(i) + '.csv'
    
    data, train_y = normalizer(fname, skip_rows, num_rows, lookback_days)
    Ytrain[i * (len(data) - lookback_days) : (i + 1) * (len(data) - lookback_days)] = train_y

    for j in range(lookback_days, len(data)):
        #first day in dataset
        first = j - lookback_days
        #get lookback data
        chunk = data[first : j]
        #calculate standard deviation
        std = np.std(chunk)
        #calculate mean
        mean = np.mean(chunk)
        amp = np.amax(chunk) - np.amin(chunk)
        delta1 = data[j][0] - data[j - 1][0]
        delta2 = data[j - 1][0] - data[j - 2][0]
        delta3 = delta1 + delta2
        ma_20 = np.mean(data[j - 20: j])
        #update information in the training dataset
        row = [delta1, delta3, amp, std, ma_20]
        Xtrain[j - lookback_days + i * (len(data) - lookback_days)] = row

#get the feature values
def featurize(data, lookback_days):
    dt = np.zeros((data.shape[0] - lookback_days, 5))
    for j in range(lookback_days, len(data)):
        #first day
        first = j - lookback_days
        #get lookback data
        chunk = data[first : j].copy()
        #calculate standard deviation
        std = np.std(chunk)
        #calculate mean
        mean = np.mean(chunk)
        amp = np.amax(chunk) - np.amin(chunk)
        delta1 = data[j][0] - data[j - 1][0]
        delta2 = data[j-1][0] - data[j - 2][0]
        delta3 = delta1 + delta2
        ma_20 = np.mean(data[j - 20: j])
        row = [delta1, delta3, amp, std, ma_20]
        dt[j - lookback_days] = row
    return dt

test_skip_rows = 1182
test_num_rows = 507

test_fname1 = 'ML4T-292.csv'
Ynormalized1, train_y1 = normalizer(test_fname1, test_skip_rows, test_num_rows, lookback_days)
Yactual1 = Ynormalized1[lookback_days:]
ls_Yactual1 = []
for i in range(len(Yactual1)):
    ls_Yactual1.append(Yactual1[i][0])
ls_Yactual1 = np.array(ls_Yactual1)
Xtest1 = featurize(Ynormalized1, lookback_days)

#E + W = 5 + 23 = 28
test_fname2 = 'ML4T-328.csv'
Ynormalized2, train_y2 = normalizer(test_fname2, test_skip_rows, test_num_rows, lookback_days)
Yactual2 = Ynormalized2[lookback_days:]
ls_Yactual2 = []
for i in range(len(Yactual2)):
    ls_Yactual2.append(Yactual2[i][0])
ls_Yactual2 = np.array(ls_Yactual2)
Xtest2 = featurize(Ynormalized2, lookback_days)

knn = KNNLearner.KNNLearner()
knn.addEvidence(Xtrain, Ytrain)
knn_Ypredict1 = knn.query(Xtest1)
knn_Ypredict2 = knn.query(Xtest2)

corr1 = np.corrcoef(knn_Ypredict1, ls_Yactual1)[0,1]
corr2 = np.corrcoef(knn_Ypredict2, ls_Yactual2)[0,1]
rms1 = np.sqrt(np.sum(np.square(ls_Yactual1 - knn_Ypredict1)) / len(ls_Yactual1))
rms2 = np.sqrt(np.sum(np.square(ls_Yactual2 - knn_Ypredict2)) / len(ls_Yactual2))

print 'Correlation Coefficient of ML4T-292: ' + str(corr1) + ' with RMS of ' + str(rms1) + ' by KNN'
print 'Correlation Coefficient of ML4T-348: ' + str(corr2) + ' with RMS of ' + str(rms2) + ' by KNN'

pre1 = [0] * 100
for i in range(len(knn_Ypredict1)):
    pre1.append(knn_Ypredict1[i])

pre2 = [0] * 100
for i in range(len(knn_Ypredict2)):
    pre2.append(knn_Ypredict2[i])

#chart for 292.csv
k = np.arange(0, 200)
plt.title('First 200 Days of Actual Price vs Predicted Price - ML4T-292.csv')
plt.ylabel('Normalized Price')
plt.xlabel('Days')
plt.plot(k, Ynormalized1[:200], color = 'b')
plt.plot(k, pre1[:200], color = 'r')
plt.legend(['Actual Price', 'Predicted Price'])
plt.savefig('first_292.png')
plt.close()

#char for 348.csv
k = np.arange(0, 200)
plt.title('First 200 Days of Actual Price vs Predicted Price - ML4T-348.csv')
plt.ylabel('Normalized Price')
plt.xlabel('Days')
plt.plot(k, Ynormalized2[:200], color = 'b')
plt.plot(k, pre2[:200], color = 'r')
plt.legend(['Actual Price', 'Predicted Price'])
plt.savefig('first_348.png')
plt.close()

#chart for 292.csv last 200 days
k = np.arange(0, 200)
plt.title('Last 200 Days of Actual Price vs Predicted Price - ML4T-292.csv')
plt.ylabel('Normalized Price')
plt.xlabel('Days')
plt.plot(k, Ynormalized1[-200:], color = 'b')
plt.plot(k, pre1[-200:], color = 'r')
plt.legend(['Actual Price', 'Predicted Price'])
plt.savefig('last_292.png')
plt.close()

#char for 348.csv last 200 days
k = np.arange(0, 200)
plt.title('Last 200 Days of Actual Price vs Predicted Price - ML4T-348.csv')
plt.ylabel('Normalized Price')
plt.xlabel('Days')
plt.plot(k, Ynormalized2[-200:], color = 'b')
plt.plot(k, pre2[-200:], color = 'r')
plt.legend(['Actual Price', 'Predicted Price'])
plt.savefig('last_348.png')
plt.close()

#scatterplot for 292.csv
k = np.arange(0, len(pre1))
plt.title('Actual Price vs Predicted Price - ML4T-292.csv')
plt.ylabel('Normalized Price')
plt.xlabel('Days')
plt.scatter(k, Ynormalized1, color = 'b')
plt.scatter(k, pre1, color = 'r')
plt.legend(['Actual Price', 'Predicted Price'])
plt.savefig('scatter_292.png')
plt.close()

#scatterplot for 348.csv
k = np.arange(0, len(pre2))
plt.title('Actual Price vs Predicted Price - ML4T-348.csv')
plt.ylabel('Normalized Price')
plt.xlabel('Days')
plt.scatter(k, Ynormalized2, color = 'b')
plt.scatter(k, pre2, color = 'r')
plt.legend(['Actual Price', 'Predicted Price'])
plt.savefig('scatter_348.png')
plt.close()

#five features for 292.csv
f1 = [0] * 100
f2 = [0] * 100
f3 = [0] * 100
f4 = [0] * 100
f5 = [0] * 100
for i in range(100):
    f1.append(Xtest1[i, 0])
    f2.append(Xtest1[i, 1])
    f3.append(Xtest1[i, 2])
    f4.append(Xtest1[i, 3])
    f5.append(Xtest1[i, 4])

k = np.arange(0, 200)
plt.title('Five Features in the First 200 Days - ML4T-292.csv')
plt.xlabel('Days')
plt.plot(k, f1)
plt.plot(k, f2)
plt.plot(k, f3)
plt.plot(k, f4)
plt.plot(k, f5)
plt.legend(['Price Change since Yesterday', 'Sum of Price Change since Two Days Ago', 'Amplitude', 'Standard Deviation', 'Moving Average of Last 20 Days'])
plt.savefig('features.png')
plt.close()