from all_imports import *

from processdata import prepare_input_transformer

class TrajectoryDataset(data.Dataset):
    
    def __init__(self,file_no,mode,submode):
        super().__init__()
        self.data,self.labels=prepare_input_transformer(file_no,mode,submode)
        self.size=len(self.data)
        self.mode=mode
        self.details=[file_no,mode,submode]

    def __len__(self):
        return self.size
    def __getitem__(self,idx):
        ipt_data=self.data[idx]
        lab=self.labels[idx]
        #print(lab,',')

        return ipt_data,lab

    

