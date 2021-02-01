import tensorflow as tf
import numpy as np
import cv2


class Covidnet():
    def __init__(self, model_path):
        # The file path of model
        self.model_path = model_path

        # Initialize the model
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.load_graph()

    def load_graph(self):
        with tf.io.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def)

        self.graph.finalize()

    def print_graph(self):
        for node in self.graph.as_graph_def().node:
            print(node.name)

    def predict(self, image):
        input_data = cv2.resize(image, (224, 224))
        input_data = input_data.astype('float32') / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        input_tensor = self.graph.get_tensor_by_name("import/input_1:0")
        output_tensor = self.graph.get_tensor_by_name("import/dense_3/Softmax:0")
        return self.sess.run(output_tensor, feed_dict={input_tensor: input_data})