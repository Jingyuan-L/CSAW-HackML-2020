import keras
from keras.optimizers import Adam
import h5py
import numpy as np
import keras.backend as K
import tensorflow as tf
import sys

DENSE_RATE_LIMIT = 0.75
ACC_DECREASE_LIMIT = 50

test_dir = str(sys.argv[1])
badNet = keras.models.load_model(str(sys.argv[2]))


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def cal_accuracy(x_test, y_test, model):
    x_test = x_test / 255
    clean_label_p = np.argmax(model.predict(x_test), axis=1)

    class_accu = np.mean(np.equal(clean_label_p, y_test)) * 100

    return class_accu


# badNet1 = keras.models.load_model('models/sunglasses_bd_net.h5')
# badNet1.load_weights('models/sunglasses_bd_weights.h5')

# badNet2 = keras.models.load_model('models/anonymous_bd_net.h5')
# badNet2.load_weights('models/anonymous_bd_weights.h5')

# test_dir = 'data/clean_test_data.h5'
validation_dir = 'data/clean_validation_data.h5'
# sunglass_dir = 'data/sunglasses_poisoned_data.h5'

test_data_x, test_data_y = data_loader(test_dir)
validation_data_x, validation_data_y = data_loader(validation_dir)
# sunglass_data_x, sunglass_data_y = data_loader(sunglass_dir)


def pruning(model, layer_name):
    print('* Start pruning *')

    # extract the output of conv_3 layer
    conv_3_output = model.get_layer(layer_name).output
    temp_model = keras.models.Model(inputs=model.input, outputs=conv_3_output)

    # calculte the contribution of each channel on the validation data
    print('Calculting the contribution of each channel on the validation data ... ')
    ############### IMPORTANT! Normalized the training data! ###################
    normalized_x = validation_data_x / 255
    conv_3_contribution = temp_model.predict(normalized_x)
    conv_3_output_len = np.shape(conv_3_contribution)[-1]
    sample_num = np.shape(conv_3_contribution)[0]
    print(np.shape(conv_3_contribution))

    each_channel_contrib = np.zeros((1, conv_3_output_len))
    for i in range(conv_3_output_len):
        cur_ave = np.mean(conv_3_contribution[:, :, :, i])
        each_channel_contrib[0, i] = cur_ave
    print('The contribution of each channel:')
    print(each_channel_contrib)

    # sort the contrbutions and find the pruning order
    weights, bias = model.get_layer(layer_name).get_weights()
    ascending_contrib = np.argsort(each_channel_contrib)
    # print(ascending_sum)
    print('The index of channels in the order of ascending contribution:')
    print(ascending_contrib)

    # pruning by set the weight and bias of the pruned channel to ZERO
    original_acc = cal_accuracy(validation_data_x, validation_data_y, model)
    prune_num = 0
    prune_limit = int(len(ascending_contrib[0]) * DENSE_RATE_LIMIT)
    for i in range(prune_limit):
        if i > 30:
            cur_acc = cal_accuracy(validation_data_x, validation_data_y, model)
            if abs(original_acc - cur_acc > ACC_DECREASE_LIMIT):
                prune_num = i
                break

        weights[:, :, :, ascending_contrib[0][i]] = np.zeros(np.shape(weights[:, :, :, ascending_contrib[0][i]]))
        bias[ascending_contrib[0][i]] = 0
        model.get_layer(name=layer_name).set_weights((weights, bias))
        prune_num = i

    print('* Finish pruning, pruned channels number: ', prune_num + 1)

    return model


def fine_tune(base_model, retrain_x, retrain_y):
    print('* Start fine tune *')
    ############### IMPORTANT! Normalized the training data! ###################
    retrain_x = retrain_x / 255
    ###########################################################################
    base_model.compile(Adam(lr=1e-3, amsgrad=True), loss="categorical_crossentropy", metrics=["accuracy"])
    validation_data_y_onehot = keras.utils.to_categorical(retrain_y, num_classes=1283)
    base_model.fit(retrain_x, validation_data_y_onehot, epochs=5, batch_size=128 * 2)
    print('* Finish fine tune *')

    return base_model


################################### NEW NET #####################################
def GoodNet(repairedNet, badNet):
    # define input
    x = keras.Input(shape=(55, 47, 3), name='input')

    repaired_y = repairedNet(x)
    poisoned_y = badNet(x)

    def check_class(combined_y):
        clean_y = combined_y[0]
        dirty_y = combined_y[1]
        # sample_num = clean_y.shape[0]
        backdoor_class = clean_y.shape[-1] + 1

        sign = tf.abs(tf.sign(K.argmax(clean_y,axis=1) - K.argmax(dirty_y,axis=1)))
        # print(sign.shape)
        y_reclassify = sign * (backdoor_class - 1) + (tf.ones_like(sign)-sign) *  K.argmax(clean_y,axis=1)
        # print(y_reclassify.shape)
        y_reclassify = tf.one_hot(y_reclassify,backdoor_class)
        # print(y_reclassify.shape)

        return y_reclassify

    y_reclassify = keras.layers.Lambda(check_class, name= 'y_reclasify')([repaired_y, poisoned_y])

    model = keras.Model(inputs=x, outputs=y_reclassify, name='goodNet')

    return model


if __name__ == '__main__':
    # print(badNet1.summary())
    print('BadNet test data classification accuracy:', cal_accuracy(test_data_x, test_data_y, badNet))
    prunedNet = pruning(badNet, 'conv_3')
    print('PrunedNet test data classification accuracy:', cal_accuracy(test_data_x, test_data_y, prunedNet))
    # print('Validation data accuracy:', cal_accuracy(validation_data_x, validation_data_y, prunedNet))
    # print('Poisoned data accuracy:', cal_accuracy(sunglass_data_x, sunglass_data_y, prunedNet))
    # print('')

    tunedNet = fine_tune(prunedNet, validation_data_x, validation_data_y)
    print('TunedNet test data classification accuracy:', cal_accuracy(test_data_x, test_data_y, tunedNet))
    # print('Validation data accuracy:', cal_accuracy(validation_data_x, validation_data_y, tunedNet))
    # print('Poisoned data accuracy:', cal_accuracy(sunglass_data_x, sunglass_data_y, tunedNet))

    print('Net has been repaired! ')

    print('Creating the new GoodNet ... ')
    # K.clear_session()
    goodNet = GoodNet(tunedNet, badNet)
    # goodNet.name = 'goodNet'
    # print(np.argmax(goodNet.predict(validation_data_x)[0:10,:],axis=1))
    # print(np.argmax(badNet1.predict(validation_data_x)[0:10,:],axis=1))
    print('GoodNet test data classification accuracy:', cal_accuracy(test_data_x, test_data_y, goodNet))
    # print('Validation data accuracy:', cal_accuracy(validation_data_x, validation_data_y, goodNet))
    # print('Poisoned data accuracy:', cal_accuracy(sunglass_data_x, sunglass_data_y, goodNet))

    goodNet.save('models/goodNet.h5')
    goodNet.save_weights('models/goodNet_weights.h5')
    print("GoodNet and its weights has been saved to 'models/goodNet.h5' and 'models/goodNet_weights.h5'! ")
