import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, Activation
from tensorflow.keras.layers import Reshape, LSTM, TimeDistributed, Bidirectional, BatchNormalization
from tensorflow.keras.models import load_model


def featurenet():
    activation = tf.nn.relu
    # activation = tf.nn.leaky_relu
    padding = 'same'

    ######### Input ########
    input_signal = Input(shape=(30*100,1), name='input_signal')
    # print("input_signal:",input_signal.shape)

    ######### CNNs with small filter size at the first layer #########
    # print("\nCNN1")
    cnn0 = Conv1D(
        kernel_size=50,
        filters=64,
        strides=6,kernel_regularizer=keras.regularizers.l2(0.001)) 
    s = cnn0(input_signal)
    s = BatchNormalization()(s) 
    s = Activation(activation=activation)(s)
    # print("cnn0:",s.shape)
    cnn1 = MaxPool1D(pool_size=8, strides=8)
    s = cnn1(s)
    # print("cnn1:",s.shape)
    cnn2 = Dropout(0.5)
    s = cnn2(s)
    # print("cnn2:",s.shape)
    cnn3 = Conv1D(kernel_size=8,filters=128,strides=1,padding=padding)
    s = cnn3(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    # print("cnn3:",s.shape)
    cnn4 = Conv1D(kernel_size=8,filters=128,strides=1,padding=padding)
    s = cnn4(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    # print("cnn4:",s.shape)
    cnn5 = Conv1D(kernel_size=8,filters=128,strides=1,padding=padding)
    s = cnn5(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    # print("cnn5:",s.shape)
    cnn6 = MaxPool1D(pool_size=4,strides=4)
    s = cnn6(s)
    # print("cnn6:",s.shape)
    cnn7 = Reshape((int(s.shape[1])*int(s.shape[2]),)) # Flatten
    s = cnn7(s)
    # print("cnn7:",s.shape)

    ######### CNNs with large filter size at the first layer #########
    # print('\nCNN2')
    cnn8 = Conv1D(
        kernel_size=400,
        filters=64,strides=50,kernel_regularizer=keras.regularizers.l2(0.001))
    l = cnn8(input_signal)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    # print("cnn8:",l.shape)
    cnn9 = MaxPool1D(pool_size=4, strides=4)
    l = cnn9(l)
    # print("cnn9:",l.shape)
    cnn10 = Dropout(0.5)
    l = cnn10(l)
    # print("cnn10:",l.shape)
    cnn11 = Conv1D(kernel_size=6,filters=128,strides=1,padding=padding)
    l = cnn11(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    # print("cnn11:",l.shape)
    cnn12 = Conv1D(kernel_size=6,filters=128,strides=1,padding=padding)
    l = cnn12(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    # print("cnn12:",l.shape)
    cnn13 = Conv1D(kernel_size=6,filters=128,strides=1,padding=padding)
    l = cnn13(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    # print("cnn13:",l.shape)
    cnn14 = MaxPool1D(pool_size=2,strides=2)
    l = cnn14(l)
    # print("cnn14:",l.shape)
    cnn15 = Reshape((int(l.shape[1])*int(l.shape[2]),))
    l = cnn15(l)
    # print("cnn15:",l.shape)

    # print('\nMERGED')
    merged = keras.layers.concatenate([s, l])
    # print("merged:",merged.shape)
    merged = Dense(1024)(merged) 
    merged = Dropout(0.5)(merged) 
    merged = Dense(5,name='merged')(merged)
    # print('merged',merged.shape)
    pre_softmax = Activation(activation='softmax')(merged)
    # print('pre_softmax',pre_softmax.shape)

    pre_model = Model(input_signal,pre_softmax)
    pre_opt = keras.optimizers.Adam(lr=1e-4)
    pre_model.compile(optimizer=pre_opt,loss='categorical_crossentropy',metrics=['acc'])

    return pre_model

def deepsleepnet(pre_model):
    input_signal = pre_model.get_layer(name='input_signal').input
    merged = pre_model.get_layer(name='merged').output

    activation_seq = 'relu'

    cnn_part = Model(input_signal, merged) # pre train 된 부분

    input_seq = Input(shape=(None, 3000, 1)) # sequence 길이 모르므로 None
    # print('input_seq', input_seq.shape)

    signal_sequence = TimeDistributed(cnn_part)(input_seq) # TimeDistributed 로 시퀀스를 입력 받을 수 있음
    # print('signal_sequence',signal_sequence.shape)

    bidirection = Bidirectional(LSTM(512, dropout=0.5,activation=activation_seq, return_sequences=True),merge_mode='concat')(signal_sequence)
    # print('bidirection',bidirection.shape)

    fc1024 = Dense(1024)(signal_sequence)
    fc1024 = BatchNormalization()(fc1024)
    fc1024 = Activation(activation=activation_seq)(fc1024)
    # print('fc1024',fc1024.shape)
    residual = keras.layers.add([bidirection, fc1024]) # skip-connection
    residual = Dropout(0.5)(residual)
    # print('residual',residual.shape)

    dense_seq = Dense(5)(residual)
    # print('dense_seq',dense_seq.shape)

    seq_softmax = Activation(activation='softmax')(dense_seq)
    # print('seq_softmax',seq_softmax.shape)

    seq_model = Model(input_seq, seq_softmax)
    seq_opt = keras.optimizers.Adam(lr=1e-6)
    seq_model.compile(loss='categorical_crossentropy', optimizer=seq_opt, metrics=['acc'])

    return seq_model

if __name__=='__main__':
    pre_model = featurenet()    
    seq_model = deepsleepnet(pre_model)

