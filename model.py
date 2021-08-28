import tensorflow as tf
# from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.python.keras.applications.vgg19 import VGG19

# == Use 'VGG19' invoke the pre-defined NN architecture, then use 'weights' specified the trained weights
model = VGG19(weights='weights/my_model.h5')


def process_image(image):
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)

    return image


def predict_class(image):
    '''
    Predict and render the class of a given image 
    '''
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # return the classification
    prediction = label[1]
    percentage = '%.2f%%' % (label[2] * 100)

    return prediction, percentage


if __name__ == '__main__':
    ''' for test'''
    # load an image from file
    image = load_img('images/elephant.png', target_size=(224, 224))
    image = process_image(image)
    prediction, percentage = predict_class(image)
    print(prediction, percentage)
