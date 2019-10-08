import tkinter
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

df = pd.read_csv('data_model_nta.csv', delimiter=',', index_col=0)

df['fi population '] = df['fi population ']/df['Total population']
X = df.iloc[:, 1:6]
Y = df.iloc[:, 6:10]

scaler = MinMaxScaler()
scaler.fit(X)
X_norm = scaler.transform(X)

scaler2 = MinMaxScaler()
scaler2.fit(Y)
Y_norm = scaler2.transform(Y)


X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = train_test_split(X_norm, Y_norm, test_size=0.1)

'''
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
Y_test = Y_test.values
Y_train = Y_train.values

reg = linear_model.LassoLars(alpha=.5, max_iter=20000, eps=0.1, verbose=1)

reg.fit(X_train, Y_train)
result = reg.predict(X_test)
accuracy_score(Y_test, result)
'''

'''
dtrain = xgb.DMatrix(X_train, label=Y_train.iloc[:, 1])
dtest = xgb.DMatrix(X_test, label=Y_test.iloc[:, 1])
evallist = [(dtest, 'eval'), (dtrain, 'train')]

param_final = {'max_depth': 20, 'eta': 0.1, 'silent': 0, 'objective': 'reg:linear',
               'gamma': 5, 'nthread': 4, 'eval_metric': 'rmse', 'lambda': 0.01, 'tree_method': 'exact'}
num_train = 2000
bst = xgb.train(param_final, dtrain, num_train, evallist)

result = bst.predict(dtest)
'''


'''
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(32, activation=tf.nn.relu,
                           input_shape=(X_train_norm.shape[1],)),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(3,)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.00001)  # 0.0001

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return model


model = build_model()
model.summary()
EPOCHS = 150

history = model.fit(X_train_norm, Y_train_norm, epochs=EPOCHS,
                    validation_split=0.2)
'''


clf = linear_model.Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=70000,
                         normalize=False, positive=False, precompute=False, random_state=None,
                         selection='cyclic', tol=0.000001, warm_start=False)
clf.fit(X_train_norm, Y_train_norm)

'''
input_array = np.arange(4)
input_array = (input_array+1)/10

final_df = pd.DataFrame()

for i in input_array:
    for j in input_array:
        for k in input_array:
            for l in input_array:
                for m in input_array:
                    result_df = pd.DataFrame()
                    inputs = np.array([i, j, k, l, m])
                    print(inputs)
                    inputs = inputs.reshape(1,-1)
                    result = clf.predict(inputs)
                    result = scaler2.inverse_transform(result)
                    inputs = scaler.inverse_transform(inputs)
                    entry = np.concatenate((inputs, result), axis=1)
                    print(entry)
                    entry_df = pd.DataFrame(entry)
                    final_df = final_df.append(entry_df)
'''


a, b, coefs = linear_model.lars_path(X_train_norm, Y_train_norm[:,1], method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()


import pickle
filename = 'finalized_model3.sav'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label='Val loss')
    plt.legend()

plot_history(history)

model.save_weights('./model_1/checkpoints')

model_2 = build_model()
model_2.load_weights('./model_1/checkpoints')

result = model.predict(X_test_norm)
result = scaler2.inverse_transform(result)

fg = plt.plot(result[:,0])
fg = plt.plot(Y_test.iloc[:,0])

fg2 = plt.plot(result[:, 1])
fg2 = plt.plot(Y_test.iloc[:, 1])


inputs = np.array[0, 0, 0, 0, 0]
inputs = inputs.reshape(1, -1)
result = loaded_model.fit(inputs)

