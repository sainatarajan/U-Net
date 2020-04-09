# -*- coding:utf-8 -*-

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import array_to_img
import cv2
from data import *


class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print(conv1)
        conv1 = BatchNormalization()(conv1)
        print ("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        print ("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print ("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print ("conv2 shape:", conv2.shape)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print ("conv2 shape:", conv2.shape)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print ("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print ("conv3 shape:", conv3.shape)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print ("conv3 shape:", conv3.shape)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print ("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        up6 = BatchNormalization()(up6)
        merge6 = concatenate([drop4, up6], axis=3)
        print(up6)
        print(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        print(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        print(conv6)
        conv6 = BatchNormalization()(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        up7 = BatchNormalization()(up7)
        merge7 = concatenate([conv3, up7], axis=3)
        print(up7)
        print(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        print(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        print(conv7)
        conv7 = BatchNormalization()(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        up8 = BatchNormalization()(up8)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        up9 = BatchNormalization()(up9)
        merge9 = concatenate([conv1, up9], axis=3)
        print(up9)
        print(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        print(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        print(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        print ("conv9 shape:", conv9.shape)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        print(conv10)
        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=100, verbose=1,
                  validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        model.save_weights('./unet_model.hdf5')
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('./data/results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('./data/results/imgs_mask_test.npy')
        piclist = []
        for line in open("./data/results/pic.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        print(len(piclist))
        for i in range(imgs.shape[0]):
            path = "./data/results/" + piclist[i]
            img = imgs[i]
            img = array_to_img(img)
            img.save(path)
            cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            cv_pic = cv2.resize(cv_pic,(1918,1280),interpolation=cv2.INTER_CUBIC)
            binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(path, cv_save)
            
    def load_model_weights(self, model):
        model.load_weights('./unet_model.hdf5')


if __name__ == '__main__':
    myunet = myUnet()
    model = myunet.get_unet()
    # model.summary()
    # plot_model(model, to_file='model.png')
    # Uncomment the below line if you want to re-train a previously trained model 
    # myunet.load_model_weights(model)
    myunet.train()
    myunet.save_img()
