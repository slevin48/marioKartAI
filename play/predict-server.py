import sys, time, logging, os, argparse

import numpy as np
from PIL import Image, ImageGrab
from socketserver import TCPServer, StreamRequestHandler

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras import optimizers
from keras import backend as K

# from keras.layers.normalization import BatchNormalization

OUT_SHAPE = 5

INPUT_WIDTH = 200
INPUT_HEIGHT = 66
INPUT_CHANNELS = 3

INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# from train import create_model, is_valid_track_code, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS

def create_model(keep_prob = 0.8):
    model = Sequential()

    # NVIDIA's model
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape= INPUT_SHAPE))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(OUT_SHAPE, activation='softsign'))

    return model


def prepare_image(im):
    im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr

class TCPHandler(StreamRequestHandler):
    def handle(self):

        logger.info("Handling a new connection...")
        for line in self.rfile:
            message = str(line.strip(),'utf-8')
            logger.debug(message)

            if message.startswith("PREDICTFROMCLIPBOARD"):
                try:
                    im = ImageGrab.grabclipboard()
                except:
                    print("failed to open clipboard")
                if im != None:
                    prediction = model.predict(prepare_image(im), batch_size=1)[0]
                    self.wfile.write((str(prediction[0]) + "\n").encode('utf-8'))
                else:
                    self.wfile.write("PREDICTIONERROR\n".encode('utf-8'))

            if message.startswith("PREDICT:"):
                im = Image.open(message[8:])
                prediction = model.predict(prepare_image(im), batch_size=1)[0]
                self.wfile.write((str(prediction[0]) + "\n").encode('utf-8'))

if __name__ == "__main__":
    logger.info("Loading model...")
    model = create_model(keep_prob=1)
    
    model.load_weights('models/model_weights_2021-10-11-1_100epochs.h5')

    logger.info("Starting server...")
    server = TCPServer(('0.0.0.0', 36296), TCPHandler)

    print("Listening on Port: {}".format(server.server_address[1]))
    sys.stdout.flush()
    server.serve_forever()
