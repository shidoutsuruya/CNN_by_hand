import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers,backend
import matplotlib.pyplot as plt
from cifar10 import x_train,x_test,y_train,y_test,onehot,accurate_curve
