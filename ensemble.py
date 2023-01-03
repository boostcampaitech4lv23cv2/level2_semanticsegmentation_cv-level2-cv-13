import os
import pandas as pd
import statistics
from tqdm import tqdm
location="/opt/ml/ensemble"
if not os.path.isdir(location):
    os.makedirs(location)

csvs=[]
for file in os.listdir(location):
    csvs.append(pd.read_csv(os.path.join(location,file)))
assert len(csvs),"Please put at least two csv files in the ensemble folder."
l=len(csvs[0])
output=pd.read_csv("/opt/ml/input/code/submission/sample_submission.csv")
for i in tqdm(range(l)):
    name=csvs[0].iloc[i]["image_id"]
    masks=[csvs[k].iloc[i]["PredictionString"].split() for k in range(len(csvs))]
    labels= list(zip(*masks))
    new_mask=[]
    for j in range(256*256):
        new_mask.append(statistics.mode(labels[j]))
    output=output.append({"image_id" : name, "PredictionString" : ' '.join(new_mask)}, 
                                   ignore_index=True)
output.to_csv(os.path.join(location,"ensemble.csv"), index=False)