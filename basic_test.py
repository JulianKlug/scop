from efficientnet_pytorch_3d import EfficientNet3D
import torch
import numpy as np
import pandas as pd

model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 6}, in_channels=4)
dataset_path = '/Users/jk1/temp/joining_test/small_rescaled_data_set.npz'
outcome_path = '/Users/jk1/temp/joining_test/joined_anon_outcomes_2015_2016_2017_2018_df.xlsx'

from torchsummary import summary
summary(model, input_size=(4, 79, 95, 79))

# model = model.cuda()
# inputs = torch.randn((1, 1, 200, 200, 200)).cuda()
# labels = torch.tensor([0]).cuda()

inputs = torch.randn((1, 1, 200, 200, 200))

model = model
raw_images = np.load(dataset_path, allow_pickle=True)['ct_inputs']
ids = np.load(dataset_path, allow_pickle=True)['ids']
inputs = torch.tensor(raw_images).permute(0, 4, 1, 2, 3).to(torch.float32)
raw_label_df = pd.read_excel(outcome_path)
raw_labels = [raw_label_df.loc[raw_label_df['anonymised_id'] == subj_id, 'combined_mRS_90_days'].iloc[0] for subj_id in ids]
labels = torch.tensor(raw_labels).long()
# test forward
num_classes = 6

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
for epoch in range(2):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    print('[%d] loss: %.3f' % (epoch + 1, loss.item()))

print('Finished Training')