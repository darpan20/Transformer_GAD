
from data_loader import TrajectoryDataset
from all_imports import *
import all_imports as ai
from statistics import stdev,mean,median
from train_model import train
from ruamel.yaml import YAML
from pathlib import Path
import numpy as np

#------------------------------------------START---------------------------#
def main(**kwargs):
   
   return train(**kwargs)

if __name__=='__main__':
     
    ''' #----------------------AVAILABLE DATASETS-----------------------#

         files=[ 'Trajectorys2_100_100_anom.csv' ,
                'Trajectorys2_100_50_anom.csv'   ,
                'Trajectorys2_100_75_anom.csv',
                'Trajectorys2_100_55_norm.csv',
                'Trajectorys2_100_85_norm.csv',
                'Trajectorys2_100_400_anom.csv',
                'rq2/Trajectorys2_100_150_anom0.0.csv',
                'rq2/Trajectorys2_100_150_anom0.2.csv',
                'rq2/Trajectorys2_100_150_anom0.5.csv',
                'rq2/Trajectorys2_100_150_anom1.0.csv' ,
                'rq2/Trajectorys2_100_150_anom0.0novelty.csv',
                 'rq2/Trajectorys2_100_150_anom0.01novelty.csv',
                 'rq2/Trajectorys2_100_150_anom0.05novelty.csv',
                 'rq2/Trajectorys2_100_150_anom0.1novelty.csv']
                                                                        '''
    

    

    ### Choose intial parameters and  pre-trained file if available ### start
    
     
    
    testing_dataset=5   # 2(75 timesteps)   or 5(400 timesteps)        #RQ-1  (all versions till 38 pre-trained)
    testing_dataset=6  #    6,7,8,9 (NOISE_level - 0,0,0.2,0.5,1.0)    #RQ-2  Noise
    #testing_dataset=10  #    10,11,12,13 (Novelty_level - 0.0,0.01,0.05,0.1)    #RQ-2  Novelty
    seeds=[34,38,30] 

    model_type='Transformers'          # 'Transformer' or 'RNN'
    #model_type='GRU_RNN'
    
    uncertainty_criteria=['mean']

    #load_pretrained_model=False
    load_pretrained_model=True  
    
    kwargs={'uncertainty_criteria':uncertainty_criteria,'model_type':model_type}
    if kwargs['model_type']=='Transformers':
            kwargs['root_dir'] =CHECKPOINT_PATH
    else:
            kwargs['root_dir'] =CHECKPOINT_PATH_RNN

    ### Choose intial parameters and  pre-trained file if available ### end
    
    if load_pretrained_model:  # Loaded trained models from Q1 experiments to test datasets for For RQ2
     
        print('\n---LOADING PRETRAINED-MODEL-----\n')

        # pre_details  as :  {seed:[version,epoch,step]} 
        # For pre-trained model path, see 'all_imports.py'
        pre_details_unsup_16={34:[3,147,444],38:[4,143,432],30:[5,53,162]}        # saved model trained with 16 heads for rq2
        pre_details_sup_16={34:[36,109,440],38:[37,108,436],30:[38,165,664]}        # saved model trained with 16 heads for rq2
        pre_details_semisup_16={34:[27,149,450],38:[28,140,423],30:[29,54,165]}          # saved model trained with 16 heads for rq2
        
        pre_details_unsup={34:[0,148,447],38:[1,149,450],30:[2,45,138]}        # saved model trained with 8 heads for rq2
        pre_details_sup={34:[12,163,656],38:[13,176,708],30:[14,156,628]}        # saved model trained with 8 heads for rq2
        pre_details_semisup={34:[24,147,444],38:[25,149,450],30:[26,45,138]}          # saved model trained with 8 heads for rq2


        pre_details=pre_details_sup      ###choose the pre_trained model details
        
        
        #print(kwargs['root_dir'])
     
        # Reading the Hparams file to get the batch_size that was used for training 
        
        hparams_path=Path(f"{kwargs['root_dir']}lightning_logs/version_{pre_details[list(pre_details.keys())[0]][0]}/hparams.yaml")
        yaml = YAML(typ='unsafe')
        with open(hparams_path, 'r') as f:
            hparams_data = yaml.load(hparams_path)
        kwargs['batch_size']=hparams_data['batch_size']
        
    else: 

        ### enter params ### start
        
        
        heads=4  # For transformer.. Set to None or any value in case of  RNN
        optimizer=optim.RAdam        
        scenarios=['sup','semisup','unsup']
        
        scenario=scenarios[1] # choose scenario

        unsup_config=[100,512,150,1e-6,heads,6,1,0.0,0.000003,80,35,uncertainty_criteria]   
        sup_config=[100,512,250,1e-5,heads,6,1,0.4,0.0003,50,25,uncertainty_criteria ]
        semisup_config=[100,512,150,1e-6,heads,6,1,0.0,0.000003,80,35,uncertainty_criteria ] 
        

        ### enter params ### end
        
        if scenario==scenarios[1]:
            selected_config=semisup_config
        elif scenario==scenarios[2]:
            selected_config=unsup_config
        else:
            
            selected_config=sup_config

        
        kwargs['input_dim']=selected_config[0]   
        kwargs['model_dim']=selected_config[1]   
        kwargs['max_epochs']=selected_config[2]
        kwargs['optim']=optimizer
        kwargs['weight_decay']=selected_config[3]
        kwargs['num_heads']=selected_config[4]
        kwargs['num_layers']=selected_config[5]         
        kwargs['num_classes']=selected_config[6]       
        kwargs['dropout']=selected_config[7]
        kwargs['lr']=selected_config[8]  
        kwargs['warmup']=selected_config[9]
        kwargs['batch_size']=selected_config[10]   
          
        # input_dim : total number of Coordinates  for each Group)
        # model_dim : Each Coordinate represented embedded to model_dim(512) in both Transformer and RNN
        # num_layers: No. of encoder layers in Transformer Encoder-Stack
        # num_classes: Final Predicted output dimension     
    
    

        print(f'MODEL TYPE:  {model_type}')
        print('Scenario:',scenario) 
        
        pre_details={}
        if model_type=='Transformers': 
            print('No. of heads',heads) 

        #load train and validation datasets for model training
        print('TRAIN DATA:----------------\n')

        if scenario==scenarios[1]:
            train_data=TrajectoryDataset(4,'train','semisup')   #semisupervised
        elif scenario==scenarios[0]:
            train_data=TrajectoryDataset(0,'train','sup')    #supervised
        else:

            train_data=TrajectoryDataset(0,'train','unsup')    #unsupervised


        print('---------------------------')


        print('VAL DATA:----------------\n')
    
        val_data=TrajectoryDataset(1,'val','sup')         # unsupervised,supervised and semisupervised
    
        kwargs['len_tr'],kwargs['len_va'],kwargs['file_tr'],kwargs['file_va']= len(train_data),len(val_data),train_data.details,val_data.details
        
       
        train_loader = data.DataLoader(train_data ,batch_size=kwargs['batch_size'], pin_memory=True)
        val_loader = data.DataLoader(val_data ,batch_size=kwargs['batch_size'],  pin_memory=True)
        kwargs['train_loader'],kwargs['val_loader']=train_loader,val_loader
        print('---------------------------')
    


    # Load test dataset for modelt testing
    print('TEST DATA:----------------\n')

    
    test_data=TrajectoryDataset(testing_dataset,'test','sup')       #sup
    kwargs['len_ts'],kwargs['file_ts']=len(test_data),test_data.details
    test_loader = data.DataLoader(test_data ,batch_size=kwargs['batch_size'],  pin_memory=True)
    kwargs['test_loader']=test_loader
    roc_auc_scores=[]      # for averaging roc_scores over different seeds
    #AAS_averages=[]      # for averaging AAS_avg over different seeds
    losses_all_val=[]      #  averaging val_losses to plot the averaged plot for doifferent seeds
    losses_all_train=[]    #  averaging train_losses to plot the averaged plot for doifferent seeds

    
    # ---- Model training loop for each seed---#

    for random_seed in seeds:

        
        kwargs['random_seed']=random_seed

        if not load_pretrained_model:
            pretrained_filename=""
            

        else:
            pretrained_filename = os.path.join(CHECKPOINT_PATH,   
            f"lightning_logs/version_{pre_details[random_seed][0]}/checkpoints",
             f"epoch={pre_details[random_seed][1]}-step={pre_details[random_seed][2]}.ckpt")
        
        kwargs['pretrained_filename']=pretrained_filename
        # metrics dict for storing along with Model_checkpoint
        kwargs['metrics']={'train_losses':[],'val_losses':[],'test_losses':[],'train_preds':[],'val_preds':[],'test_preds':[],'train_labels':[],'val_labels':[],'test_labels':[]}
        
        pl.seed_everything(random_seed)
        ai._set_seeds(random_seed)
        roc,losses_train,losses_val=main(**kwargs)
                                                
        roc_auc_scores.append(roc)
        #AAS_averages.append(AAS_avg)
        losses_all_train.append(losses_train)
        losses_all_val.append(losses_val)
    
    print(f'\n---------------FINAL RESULTS  averaged over seeds -  {seeds}--------\n')
    
    print(f'\nROC-AUC score with stdev : {mean(roc_auc_scores),np.std(roc_auc_scores, ddof=1)}\n')
    
    
    #print(f'\nAAS\n')
    #for i in range(len(AAS_averages[0])):
    #    m=list(map(lambda y:y[i], AAS_averages))
    #
    #    print(f'\n-----Criterion:  {uncertainty_criteria[i]}: {mean(m)} ,{np.std(m, ddof=1)} ')



    avg_losses_train=[]
    avg_losses_val=[]

    
    for i,j,k in zip(losses_all_val[0],losses_all_val[1],losses_all_val[2]): 
            avg_losses_val.append(mean([i,j,k]))
    for i,j,k in zip(losses_all_train[0],losses_all_train[1],losses_all_train[2]):         
            avg_losses_train.append(mean([i,j,k]))
    
    if not load_pretrained_model:
        #Plotting the Averaged Train-Val loss Plot over differnt differnet seeds
        plt.plot(range(len(avg_losses_train)), avg_losses_train, 'g', label='Training loss')
        plt.plot(range(len(avg_losses_val)), avg_losses_val, 'b', label='validation loss')
        plt.title(f'Best val-loss Plots averaged over seeds : {seeds} ')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
                                        
                                                

