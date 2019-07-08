from nn.fc_models.pfnn_np import PFNN

target_file = ("trained_models/epoch_0.json")

pfnn = PFNN.load(target_file)

out = pfnn.forward_pass([[], 0])
print(out)



