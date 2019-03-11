import os
from os.path import join
from deepsleepnet import featurenet, deepsleepnet
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import Loader
import pickle
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_acc(y_pred, y_test):
    cnt = 0
    for i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == np.argmax(y_test[i]):
            cnt += 1
    acc = float(cnt)/len(y_pred)
    print("Pre-train ACC:", acc)
    return "ACC: "+str(acc)
    
def get_seq_acc(y_seq_pred, y_seq_test):
    cnt = 0
    cnt2 = 0
    for i in range(len(y_seq_pred)):
        for j in range(len(y_seq_pred[i])):
            cnt2 += 1
            if np.argmax(y_seq_pred[i][j]) == np.argmax(y_seq_test[i][j]):
                cnt += 1
    acc = float(cnt) / cnt2
    print("Finetune ACC:", acc)
    return "ACC: "+str(acc)

if __name__=='__main__':
    path = ('./weights', './history')
    for p in path:
        if os.path.exists(p) is False:
            os.mkdir(p)

    VERBOSE = 0
    early_stopping = EarlyStopping(patience=20)

    PRE_EPOCH = 100
    PRE_BATCH = 200

    SEQ_EPOCH = 100
    SEQ_BATCH = 10


    for i in range(4,20,1): 
        print(i,'-th fold out of 20-fold cross validation')
        d = Loader()
        # pre training
        d.load_pretrain(i)
        pre_model = featurenet()
        pre_history = pre_model.fit(
                d.X_train,
                d.y_train,
                batch_size=PRE_BATCH,
                epochs=PRE_EPOCH,
                verbose=VERBOSE,
                validation_data=(d.X_valid, d.y_valid),
                callbacks=[early_stopping]
            )

        get_acc(pre_model.predict(d.X_test),d.y_test)

        pre_model.save(join(path[0],'pre_model_'+str(i)+'.weights'))

        with open(join(path[1],'pre_history_'+str(i)+'.bin'),'wb') as f:
            pickle.dump(pre_history.history, f)

        del pre_history

        # fine tuning
        d.load_finetune(i)
        seq_model = deepsleepnet(pre_model)
        seq_history = seq_model.fit(
            d.X_seq_train,
            d.y_seq_train,
            batch_size=SEQ_BATCH,
            epochs=SEQ_EPOCH,
            verbose=VERBOSE,
            validation_data=(d.X_seq_valid,d.y_seq_valid),
            callbacks=[early_stopping]
            )

        get_seq_acc(seq_model.predict(d.X_seq_test),d.y_seq_test)

        seq_model.save(join(path[0],'seq_model_'+str(i)+'.weights'))
        with open(join(path[1],'seq_history_'+str(i)+'.bin'),'wb') as f:
            pickle.dump(seq_history.history,f)

        del pre_model, seq_model, d

        print()
