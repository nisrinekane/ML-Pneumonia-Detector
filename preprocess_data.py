import pandas as pd
from sklearn.model_selection import train_test_split

# read csv file
data = pd.read_csv('CXR8/LongTailCXR/nih-cxr-lt_single-label_train.csv')
# filter data to only include "no finding" and "pneumonia" labels
data = data[(data['No Finding'] == 1) | (data['Pneumonia'] == 1)]
# create a new column for labels
data['Label'] = data.apply(lambda row: 'No Finding' if row['No Finding'] == 1 else 'Pneumonia', axis=1)
# drop unnecessary columns
columns_to_drop = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothrax', 'Pneumoperitoneum', 'Pneumomediastinum', 'Subcutaneous Emphysema', 'Tortuous Aorta', 'Calcification of the Aorta', 'No Finding', 'subject_id']
# use id for image name
image_column = 'id'

# split data into train, test and vali
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

# save split data to csv files
train_data.to_csv('train.csv', index=False)
val_data.to_csv('val.csv', index=False)
test_data.to_csv('test.csv', index=False)


