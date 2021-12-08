import os
import shutil
import sys
import unittest

import tensorflow as tf

import test_utility

sys.path.append(os.path.split(os.path.dirname(__file__))[0])

model_path = os.path.join(os.path.dirname(__file__), "temp", "models")


def create_model(const_factor=1, const_summand=0):
    inp = tf.keras.layers.Input(1)
    layer = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x) * const_factor + const_summand)
    x = layer(inp)
    model = tf.keras.models.Model(inputs=[inp], outputs=[x])
    return model


class test_existing_model_layer(test_utility.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.creator = test_utility.ModelFromTestJsonCreator(__file__)

    def load_assert(self, ext_model_name, int_model_id):
        summand = 1
        factor = 3
        x = 10
        result = x * factor + summand

        # Create external model
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        external_model = create_model(factor, summand)
        external_model.save(os.path.join(model_path, ext_model_name))

        # Test integration of the external model
        model = self.creator.load_model_by_id(int_model_id)
        prediction = model.predict([x])
        self.assertClose(result, prediction)

        # Test saving and loading of the hybrid model
        hybrid_model_path = os.path.join(model_path, int_model_id)
        model.save_to_file(hybrid_model_path)
        model_description = self.creator.data[int_model_id]

        loaded_model = self.creator.creator.load_model_from_file(model_description, hybrid_model_path)
        prediction = loaded_model.predict([x])
        self.assertClose(result, prediction)
        shutil.rmtree(model_path)

    def test_saved_model_format(self):
        self.load_assert(ext_model_name="saved_model_format_model", int_model_id="saved_model_format")

    def test_h5_format(self):
        self.load_assert(ext_model_name="h5_format_model.h5", int_model_id="h5_format")


# TODO h5 format, savedmodel format

if __name__ == "__main__":
    t = test_existing_model_layer()
    t.test_saved_model_format()
    t.test_h5_format()
    unittest.main()
