from ModelHandler import *
import numpy as np

modelhandler = ModelHandler("trained_models/1DCNN")
print(modelhandler.predict(np.ones(27500)))