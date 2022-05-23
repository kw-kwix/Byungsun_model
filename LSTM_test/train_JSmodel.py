import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflowjs as tfjs
from keras.metrics import AUC, Precision, Recall
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from operation import load_data, seperate_label, plot_graph


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
actions = ['crunch', 'lying_leg_raise', 'side_lunge', 'standing_knee_up', 'standing_side_crunch']


for _, action in enumerate(actions):
    path_dir1 = 'C:/Users/UCL7/VS_kwix/created_dataset/dataset_version9/' + str(action)
    path_dir2 = 'C:/Users/UCL7/VS_kwix/test_dataset/' + str(action)
    folder_list1 = os.listdir(path_dir1)
    folder_list2 = os.listdir(path_dir2)


    data = load_data(path_dir1, folder_list1)
    test_data = load_data(path_dir2, folder_list2)

    x_data, y_data = seperate_label(data)
    test_xdata, test_ydata = seperate_label(test_data)


    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True,
                    input_shape=x_data.shape[1:3])),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy', AUC(), Precision(), Recall()])
    print(model.summary)

    model_path = 'C:/Users/UCL7/VS_kwix/new_model' + '/' + str(action) + '_model_v9_' + str(_) + '.h5'

    history = model.fit(
        x_data,
        y_data,
        validation_split=0.1,
        epochs=10,
        callbacks=[
            ModelCheckpoint(model_path,
                            monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                            patience=10, verbose=1, mode='auto')
        ]
    )

    model = load_model(model_path)
    tfjs.converters.save_keras_model(model, 'C:/Users/UCL7/VS_kwix/js_model' + '/' + str(action) + '_model_tfjs_' + str(_))

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