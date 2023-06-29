from eureka.ml.model_client.base_model_client import BaseModelClient
import numpy as np
import cv2
import json

model_id = '0df5f41b1093488caed04fdff317b89f-torch'
image_path = '/home/archuser/switch-pose-estimation/rgb-xyz/rgb/image (1).png'
json_path = '/home/archuser/switch-pose-estimation/detected_result/image (1).json'

model_client = BaseModelClient()
loaded_models = [model_id.name for model_id in model_client.get_loaded_models()]
print(loaded_models)
if model_id not in loaded_models:
    model_client.load_model(model_id)

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
detected = model_client.do_object_detection(model_id, image)[0]

def convert_np_array_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np_array_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_np_array_to_list(item) for item in obj]
    else:
        return obj

detected = convert_np_array_to_list(detected)

with open(json_path, 'w') as f:
    json.dump(detected, f)