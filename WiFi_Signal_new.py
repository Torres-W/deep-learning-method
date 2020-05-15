import os
import pickle
import msvcrt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
from sys import exit
from pylab import rcParams
from scipy import stats
from scipy.io import loadmat
from scipy.interpolate import CubicSpline
from sklearn import metrics
from sklearn.model_selection import train_test_split

## 参数设置
# 采样点
N_TIME_STEPS = 400
N_FEATURES = 3
RANDOM_SEED = 42
# 制作segments的(.mat)数据个数
Data_Number = 100
# 标签0的个数
Number_label_0 = 60
# 标签1的个数
Number_label_1 = 40
# BUILDING THE MODEL
N_CLASSES = 2
N_HIDDEN_UNITS = 64


def make_segments():
    segments = []
    # xs = []
    # ys = []
    # zs = []

    for i in range(1, Data_Number+1):
            # 幅度乘积路径
         load_data = sio.loadmat('H:\\111\\1\\a3s{}.mat'.format(i))
         xs = load_data['a3'][0]

            # 相位差路径
         load_data = sio.loadmat('H:\\111\\2\\a2s{}.mat'.format(i))
         ys = load_data['a2'][0]

            # 幅度值路径
         load_data = sio.loadmat('H:\\111\\3\\Am1a3s{}.mat'.format(i))
         zs = load_data['Am1a3'][0]
         segments.append([xs,ys,zs])

    return segments

def make_labels():
    labels = []

    for i in range(0, Number_label_0):
        labels.append(0)

    for i in range(0, Number_label_1):
        labels.append(1)
    for i in range(0, Number_label_0):
            labels.append(0)
    for i in range(0, Number_label_1):
            labels.append(1)
    return labels

def reshape():
    print("size of segments: ", np.array(segments).shape)

    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    # np.assary将输入数据转化为矩阵数据,reshape在不改变矩阵的数值的前提下修改矩阵的形状
    print("size of reshaped segments: ", np.array(reshaped_segments).shape)

    label = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    print("size of labels: ", np.array(label).shape)

    return reshaped_segments, label

def create_LSTM_model(inputs):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))      ##字典型，随机数
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    X = tf.transpose(inputs, [1,0,2])                         ##矩阵转置
    X = tf.reshape(X, [-1, N_FEATURES])                       ##rashape需要的格式

    hidden =tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])      ##tf.matmul矩阵乘法将weight和biase放一块
    hidden =tf.split(hidden, N_TIME_STEPS, 0)

    #Stack 2 LSTM layers

    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)   ##对隐藏层的操作

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    lstm_last_output = outputs[-1]   ##状态信息
    return tf.matmul(lstm_last_output, W['output']) + biases['output']
##数据增强部分
def _generate_random_curve(x, sigma=0.2, knot=4):
    xx = ((np.arange(0, x.shape[0], (x.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    x_range = np.arange(x.shape[0])
    cs = CubicSpline(xx[:], yy[:])
    return np.array(cs(x_range)).transpose()


def _distort_timesteps(x, sigma=0.2):
    tt = _generate_random_curve(x, sigma)
    tt_cum = np.cumsum(tt, axis=0)
    t_scale = (x.shape[0] - 1) / tt_cum[-1]
    tt_cum = tt_cum * t_scale
    return tt_cum

def time_warp(x, sigma=0.2):
    output = np.zeros(x.shape)
    for i in range(x.shape[1]):
        tt_new = _distort_timesteps(x[:, i], sigma)
        tt_new = np.clip(tt_new, 0, x.shape[0] - 1)
        output[:, i] = x[tt_new.astype(int), i]
    return output


segments = make_segments()
labels = make_labels()
segments2 = np.array(segments)
segments2 = np.transpose(segments2)
segmentsa = time_warp(segments2,0.2)
segmentsa = np.transpose(segmentsa)
segmentsa = np.asarray(segmentsa, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
reshaped_segments, labels = reshape()
print('数据增强前：',reshaped_segments)
print('数据增强前：',reshaped_segments.shape)
reshaped_segments = np.vstack((reshaped_segments,segmentsa))
print('数据增强后：',reshaped_segments)
print('数据增强后：',reshaped_segments.shape)
# 分割训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)
# x：segments，y：labels


tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
Y = tf.placeholder(tf.float32, [None, N_CLASSES])

pred_Y = create_LSTM_model(X)                         # pred_Y是把X_train送入模型的返回值
pred_softmax = tf.nn.softmax(pred_Y, name="y_")       # tf.nn.softmax表示max的概率

# using L2 regularization for minimizing the loss

L2_LOSS = 0.0015
L2 = L2_LOSS * \
sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())    ##正则化防止过拟合
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels= Y)) + L2
#tf.reduce_mean是对向量求均值主要是交叉熵求，*\表示另起一行
# Defining the optimizer for the model

LEARNING_RATE = 0.001

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))   # correct_pred=argmax()函数目的找到矩阵的最大数对应的索引值。equal()相等是1不等是0

accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))     # accuracy是对correct_pred取均值（correct_pred每次覆盖怎么取均值？）

# Training the model

saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

N_EPOCHS = 300
BATCH_SIZE = 10

history = dict(train_loss = [], train_acc = [], test_loss = [], test_acc = [])
# train_count = len(X_train)


for i in range(1, N_EPOCHS + 1):                            # 1轮中通过optimizer调整loss->pred_Y(pred-softmax)->LSTM_model
    X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=0.3)
    train_count = len(X_train)
    for start, end in zip(range(0, train_count, BATCH_SIZE),
                          range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
        sess.run(optimizer, feed_dict={X: X_train[start:end],
                                       Y: y_train[start:end]})                   #使用优化器每一轮都在调整网络

    _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
        X: X_train, Y:y_train})
    _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
        X: X_test, Y:y_test})
    # input()

    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)

    print("train accuracy in history {0:f}".format(acc_train))
    print("train loss in history {0:f}".format(loss_train))
    print("test accuracy in history {0:f}".format(acc_test))
    print("test loss in history {0:f}".format(loss_test))

    # if i!=1 and i%10!=0:
    #     continue
    # print("Results:")
    #
    # print("Epoch: {0}, train accuracy: {1:f}, Loss: {2:f}".format(i, acc_train, loss_train))
    # print("Epoch: {0}, Test accuracy: {1:f}, Loss: {2:f}".format(i,acc_test,loss_test))
    predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y:y_test})
# print("Final Results: Accuracy: {0:.2f}, Loss: {1:.2f}".format(acc_final,loss_final))
#input()
# saving all the predictions and history using the pickle library & create a graph.

pickle.dump(predictions, open("predictions.p", "wb"))   # dump函数：将obj对象序列化存入已经打开的file中。
pickle.dump(history, open("history.p", "wb"))           # history，prediction结果存入指定文件
tf.train.write_graph(sess.graph_def, '.', 'har.pbtxt')  # tensorflow的模型保存
saver.save(sess, save_path= "./checkpoint/har.ckpt")    # 模型的保存
sess.close()

# Loading the files back for evaluating the trained model w.r.t to number of EPOCHS

history = pickle.load(open("history.p", "rb"))
predictions = pickle.load(open("predictions.p", "rb"))

# Evaluations: Plotting the graph

plt.figure(figsize=(12, 8))

plt.plot(np.array(history['train_loss']), "r-", label="Training Loss")
#plt.plot(np.array(history['train_acc']), "g--", label="Training Accuracy")

#plt.plot(np.array(history['test_loss']), "r-", label="Test Loss")
#plt.plot(np.array(history['train_acc']), "g-", label="Test Accuracy")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training progress(Loss or accuracy)')
plt.xlabel('Training EPOCH')
plt.ylim(0)
plt.savefig('Training iterations.png')
plt.show()

# Building the confusion matrix for display the model predictions vs actual predictions

LABELS = ['DOWNSTAIRS','JOGGING','SITTING','STANDING','UPSTAIRS','WALKING']

max_test = np.argmax(y_test, axis=1)   ###argmax返回的是最大数的索引
max_predictions = np.argmax(predictions, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)    ###混淆矩阵TPTF

plt.figure(figsize=(16,14))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("CONFUSION MATRIX : ")
plt.ylabel('True Label')
plt.xlabel('Predicted label')
plt.savefig('cmatrix.png')
plt.show();

#freeze the graph to save all the structure, graph and weights into a single protobuf file.

from tensorflow.python.tools import freeze_graph

input_graph_path = '' + 'har' + '.pbtxt'
checkpoint_path = './checkpoint/' + 'har' + '.ckpt'
restore_op_name = "save/Const:0"
output_frozen_graph_name = '' + 'har' + '.pb'

freeze_graph.freeze_graph(input_graph_path, input_saver="", input_binary=False,
                          input_checkpoint=checkpoint_path, output_node_names="y_", restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0",
                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")
#模型文件和权重文件整合合并为一个文件