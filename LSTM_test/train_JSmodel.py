import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflowjs as tfjs
from keras.metrics import AUC, Precision, Recall
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from operation import load_data, load_test_data, seperate_label, plot_graph


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
actions = ['crunch', 'lying_leg_raise', 'side_lunge', 'standing_knee_up', 'standing_side_crunch']

units = [32, 64]
dropout_rate = [0.1, 0.2]
earlystop_rate = [0.0001, 0.0005]

for _, action in enumerate(actions):
    if action == 'crunch':
        unit = units[0]
        dropout = dropout_rate[1]
        earlystop = earlystop_rate[0]

    elif action == 'lying_leg_raise':
        unit = units[0]
        dropout = dropout_rate[0]
        earlystop = earlystop_rate[1]

    elif action == 'side_lunge':
        unit = units[1]
        dropout = dropout_rate[0]
        earlystop = earlystop_rate[1]

    elif action == 'standing_knee_up':
        unit = units[1]
        dropout = dropout_rate[0]
        earlystop = earlystop_rate[0]

    else:
        unit = units[0]
        dropout = dropout_rate[0]
        earlystop = earlystop_rate[0]


    path_dir1 = 'C:/Users/UCL7/Documents/GitHub/HAR_model/train_dataset/' + str(action)         # 학습용 데이터 로드
    path_dir2 = 'C:/Users/UCL7/Documents/GitHub/HAR_model/test_dataset/' + str(action)          # 평가용 데이터 로드
    folder_list1 = os.listdir(path_dir1)
    folder_list2 = os.listdir(path_dir2)


    data = load_data(path_dir1, folder_list1)
    test_data = load_test_data(path_dir2, folder_list2)

    x_data, y_data = seperate_label(data)
    test_xdata, test_ydata = seperate_label(test_data)

    model = Sequential([
        Bidirectional(LSTM(unit, return_sequences=True,
                    input_shape=x_data.shape[1:3], dropout=dropout)),
        LSTM(128, dropout=dropout, return_sequences=True),
        LSTM(64, dropout=dropout),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy', AUC(), Precision(), Recall()])

    model_path = 'C:/Users/UCL7/Documents/GitHub/HAR_model/model_py' + '/' + str(action) + '_model.h5'
    earlystopping = EarlyStopping(
        monitor='val_loss', patience=5, min_delta=earlystop)

    history = model.fit(
        x_data,
        y_data,
        validation_split=0.1,
        epochs=50,
        callbacks=[earlystopping,
            ModelCheckpoint(filepath=model_path, 
                            monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                            patience=10, verbose=1, mode='auto')
        ]
    )

    model = load_model(model_path)
    tfjs.converters.save_keras_model(model, 'C:/Users/UCL7/Documents/GitHub/HAR_model/model_js/' + str(action) + '_model_tfjs_')

    test_results = model.evaluate(
        test_xdata, test_ydata)

    y_val_loss = history.history['val_loss']
    y_loss = history.history['loss']
    x_len = np.arange(len(y_loss))
    y_test_loss = np.full(x_len.shape, test_results[0])
    y_test_acc = np.full(x_len.shape, test_results[1])
    y_test_auc = np.full(x_len.shape, test_results[2])
    y_test_precision = np.full(x_len.shape, test_results[3])
    y_test_recall = np.full(x_len.shape, test_results[4])

    plt.figure(figsize = (9, 8))
    plot_graph(x_len, y_loss, 'Loss', 1)
    plot_graph(x_len, y_val_loss, 'Val_loss', 2)

    plt.figure(figsize = (9, 8))
    plot_graph(x_len, y_test_acc, 'ACC', 1)
    plot_graph(x_len, y_test_auc, 'AUC', 2)
    plot_graph(x_len, y_test_acc, 'PRECISION', 3)
    plot_graph(x_len, y_test_recall, 'RECALL', 4)
    plt.show()