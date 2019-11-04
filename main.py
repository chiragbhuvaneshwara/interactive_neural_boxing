
from src.controlers.directional_controller import DirectionalController
from src.nn.fc_models.pfnn_np import PFNN
from src.nn.fc_models.pfnn_tf import PFNN as PFNNTF
from src.nn.fc_models.mann_tf import MANN as MANNTF
from src.nn.fc_models.vinn_tf import VINN as VINNTF
from src.servers.simpleThriftServer.simpleThriftServer import CREATE_MOTION_SERVER
from src.servers.MultiThriftServer.MultiThriftServer import CREATE_MOTION_SERVER as CREATE_MULTI_MOTION_SERVER
import numpy as np
import json

args_dataset = "./MANN/mosi_dev_vinn/data/data_4D_60fps.json"
args_output = "./MANN/mosi_dev_vinn/trained_models/pfnn/"

with open(args_dataset) as f:
    config_store = json.load(f)
datasetnpz = args_dataset.replace(".json", ".npz")
MANNTF.from_file(datasetnpz, args_output, 10, config_store)