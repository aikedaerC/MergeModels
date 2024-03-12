from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, model, args=None):
        # TODO: Make args mandatory during initialization
        self.model = model['feature_model']
        self.classifer = model['classifer']
        self.id = 0
        self.args = args

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, device, args=None):
        # TODO: Remove args after modifying all dependent files
        pass

    @abstractmethod
    def test(self, test_data, device, args=None):
        # TODO: Remove args after modifying all dependent files
        pass

    # @abstractmethod
    # def save_checkpoint(self, filepath):
    #     """Save the model parameters to a file."""
    #     pass

    # @abstractmethod
    # def load_checkpoint(self, filepath):
    #     """Load the model parameters from a file."""
    #     pass