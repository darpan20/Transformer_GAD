

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def prepare_input_transformer(file:int=0,mode:str='train',submode:str=''):
    
    
    files=[   'Trajectorys2_100_100_anom.csv' ,
            'Trajectorys2_100_50_anom.csv'   ,
            'Trajectorys2_100_75_anom.csv',
            'Trajectorys2_100_55_norm.csv',
            'Trajectorys2_100_85_norm.csv',
            'Trajectorys2_100_400_anom.csv',
            'rq2/Trajectorys2_100_150_anom0.0.csv',
            'rq2/Trajectorys2_100_150_anom0.2.csv',
             'rq2/Trajectorys2_100_150_anom0.5.csv',
             'rq2/Trajectorys2_100_150_anom1.0.csv',

             'rq2/Trajectorys2_100_150_anom0.0novelty.csv',
            'rq2/Trajectorys2_100_150_anom0.01novelty.csv',
             'rq2/Trajectorys2_100_150_anom0.05novelty.csv',
             'rq2/Trajectorys2_100_150_anom0.1novelty.csv'

]
    file=f'datasets/{files[file]}'
    print(file)

    #random_state=40
    print(f'\n-------------Creating custom *{mode}* dataset from dataset file-----\n')

    


    df = pd.read_csv(file)

    print(df)

    labels=np.array(df['Target'])
   
    _,total_timesteps,total_trajectories,_=file.split('_')
    total_timestep,total_trajectories=int(total_timesteps),int(total_trajectories)
    
    print('Total-timesteps:',total_timestep,'Total-trajectories:',total_trajectories)
    print('Labels:',labels[:total_trajectories])
    
    
    considered_trajectories=list(set(i for i in df.Person))[:total_trajectories]


    #----------get scaled coordinates for all trajectories stacked start

    for traj in considered_trajectories:      # set(i for i in df.traj)

            df_temp=df.loc[df['Person'] == traj]
            df_temp=df_temp[['X_Coord','Y_Coord']].to_numpy()
            #df_temp=df.loc[df['Person'] == traj].to_numpy()[31:45] # selecting subset of timesteps
            try:
                coords=np.vstack((coords,df_temp))
            except:
                coords=df_temp
    
    
    mm_scaler=MinMaxScaler()           # or RobustScaler() or StandardScaler()
    mm_scaler.fit(coords)
    coords_scaled=mm_scaler.transform(coords)
    

    coords_scaled=np.reshape(coords_scaled,(len(considered_trajectories),total_timestep,coords_scaled.shape[1]))
    
    

    if submode=='unsup':
        print('\n Making all labels 0 for Unsupervised Training-\n')
        labels_unsup=np.zeros(len(labels[:total_trajectories]))
        print(labels_unsup,labels_unsup.shape)
        return coords_scaled,labels_unsup
    
    else:
        return coords_scaled,labels[:total_trajectories]
   
    


    





if __name__=='__main__':
    prepare_input_transformer()