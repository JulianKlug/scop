from efficientnet_pytorch_3d import EfficientNet3D
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=4)
dataset_path = '/home/klug/working_data/perfusion_maps/no_GT/noGT_pmaps_15-19_dataset_with_combined_mRS_90_days.npz'
outcome_path = '/home/klug/working_data/clinical/clinical_outcome/joined_anon_outcomes_2015_2016_2017_2018_df.xlsx'

from torchsummary import summary
summary(model, input_size=(4, 79, 95, 79))


model = model
raw_images = np.load(dataset_path, allow_pickle=True)['ct_inputs']
ids = np.load(dataset_path, allow_pickle=True)['ids']
raw_label_df = pd.read_excel(outcome_path)
raw_labels = [raw_label_df.loc[raw_label_df['anonymised_id'] == subj_id, 'combined_mRS_0-2_90_days'].iloc[0] for subj_id in ids]

train_inputs, test_inputs, train_labels, test_labels = train_test_split(raw_images, raw_labels, test_size=0.33, random_state=42)

train_inputs = torch.tensor(train_inputs).permute(0, 4, 1, 2, 3).to(torch.float32)
test_inputs = torch.tensor(test_inputs).permute(0, 4, 1, 2, 3).to(torch.float32)

train_labels = torch.tensor(train_labels).long()
test_labels = torch.tensor(test_labels).long()

# test forward
num_classes = 2

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
for epoch in range(200):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    # print statistics
    print('[%d] loss: %.3f' % (epoch + 1, loss.item()))

    with torch.no_grad():
        outputs = model(test_inputs)
        loss = criterion(outputs, test_labels)
        print('Validation loss: ', loss)


print('Finished Training')