import tflearn
import math
import os

def update_dict(storage_name, update_dict=None):
    d = dict()
    if os.path.exists(storage_name):
        with open(storage_name, 'r') as f:
            d = cPickle.load(f)
    if update_dict and len(update_dict) > 0:
        d.update(update_dict)
        with open(storage_name, 'w') as f:
            cPickle.dump(d, f)
    return d

class ModelSelector(tflearn.callbacks.Callback):
    def __init__(self, config, min_loss=1000):
        self.min_loss = min_loss
        self.config = config

    def on_epoch_end(self, training_state):
        if training_state.val_loss < self.min_loss:
            self.min_loss = training_state.val_loss
            update_dict(self.config, {'checkpoint' : training_state.step})