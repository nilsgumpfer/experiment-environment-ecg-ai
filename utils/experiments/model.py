import json
import logging

from tensorflow.keras.models import model_from_json

from utils.viz.model import plot_model


def save_model_based_on_thresholds(model, path, epoch, metrics, thresholds):
    ok = True

    for m in thresholds:
        if metrics[m] <= thresholds[m]:
            ok = False

    if ok:
        save_model(model, path, epoch)


def save_model(model, path, epoch):
    model_path = '{}/model.json'.format(path)
    weights_path = '{}/weights_e{}.h5'.format(path, epoch)

    model_json = model.to_json()

    with open(model_path, 'w') as json_file:
        json_file.write(model_json)

    logging.debug('Saved model to disk.')

    model.save_weights(weights_path)
    logging.debug('Saved weights of epoch {} to disk.'.format(epoch))


def load_model(path, epoch, remove_softmax=False):
    model_path = '{}/model.json'.format(path)
    weights_path = '{}/weights_e{}.h5'.format(path, epoch)

    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    if remove_softmax:
        # Convert JSONized model config to dictionary
        dct = json.loads(loaded_model_json)

        # Extract layer list
        layers = dct['config']['layers']

        # Remove Softmax layer if present
        for i, layer in enumerate(layers):
            if layer['class_name'] == 'Softmax':
                layers.pop(i)
                break

        # Convert dict back to json string
        loaded_model_json = json.dumps(dct)

        # Replace Softmax by ReLU activations if present
        loaded_model_json = loaded_model_json.replace('"activation": "softmax"', '"activation": "relu"')

    model = model_from_json(loaded_model_json)
    logging.info('Loaded and initialized model from disk.')

    model.load_weights(weights_path)
    logging.info('Loaded weights of epoch {} from disk into model.'.format(epoch))

    return model


def visualize_model(model, path):
    plot_path = '{}/model.png'.format(path)

    plot_model(model,
               to_file=plot_path,
               show_shapes=True,
               show_layer_names=False,
               expand_nested=False,
               dpi=100)

    logging.debug('Visualized model as PNG file.')


def load_model_and_weights_from_paths(modelpath, weightspath):
    with open(modelpath, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    logging.info('Loaded and initialized model from disk.')

    model.load_weights(weightspath)
    logging.info('Loaded weights from disk into model.')

    return model
