
from all_imports import *
from transformer import TrajectoryPredictor
from baseline import GRU_RNN
from sklearn.metrics import roc_auc_score,recall_score,precision_recall_curve,auc
from statistics import stdev,mean,median
import pandas as pd


def train(**kwargs):
    #print(kwargs)
    # Defining the path for loading or storing the trained model
    root_dir =kwargs['root_dir']

    pretrained_filename=kwargs['pretrained_filename']
    print(pretrained_filename)

    #------------------------------------------
    # Create a PyTorch Lightning trainer with the generation callback and earlystopping based on best-val loss
    best_loss='val_loss' 
    checkpoint_callback=ModelCheckpoint(save_weights_only=False, mode="min", monitor=best_loss)
    earlystop_callback=EarlyStopping(monitor="val_loss", patience=60, mode="min")
    os.makedirs(root_dir, exist_ok=True)

    try: 
        max_epochs=kwargs['max_epochs']
    except:
        max_epochs=None    # when pre-loaded model for testing
    
    if str(device)=='cpu':
        accelerator='cpu'
    else:
        accelerator='gpu'
        
    trainer_args={'accelerator':accelerator,
        'default_root_dir':root_dir,
        'callbacks':[checkpoint_callback,earlystop_callback],
        'devices':1 if str(device).startswith("cuda") else None,
        'max_epochs':max_epochs,
        'gradient_clip_val':5
        }
    trainer = pl.Trainer(**trainer_args)
    

    print(pretrained_filename)
    if os.path.isfile(pretrained_filename):  #Load pretrained model if available
        print("Found pretrained model, loading...")
        
        plots_of='best_epoch'
        if kwargs['model_type']=='Transformers':

            model = TrajectoryPredictor.load_from_checkpoint(pretrained_filename)
        else:
            model = GRU_RNN.load_from_checkpoint(pretrained_filename)
        
   
    else:  

        if kwargs['model_type']=='Transformers':
            model = TrajectoryPredictor(max_iters=trainer.max_epochs * len(kwargs['train_loader']), **{i:kwargs[i] for i in kwargs if i not in ['root_dir','dataset','train_loader','val_loader','test_loader']})
        else:
            model = GRU_RNN(max_iters=trainer.max_epochs * len(kwargs['train_loader']), **{i:kwargs[i] for i in kwargs if i not in ['root_dir','num_layers','num_heads','uncertainty_criteria','dataset','train_loader','val_loader','test_loader']})

        
        trainer.fit(model, kwargs['train_loader'],kwargs['val_loader'])
    
    #---------------------------------------------------
    # Getting the Predictions and Lables 

    if pretrained_filename !="":  ### when the predictions done using pre-trained model
        test_labels=[]
        test_result = trainer.predict(model, kwargs['test_loader'])
        for data,lab in kwargs['test_loader']:
    
            test_labels.extend(list(lab.cpu().detach().numpy()))
            

        test_preds=[]
        for i in test_result:
            test_preds.extend(i[0].cpu().detach().numpy().flatten()) 
        metrics=model.metrics
        best_val_loss_index=pd.Series(metrics['val_losses']).idxmin()

    else:

        test_result = trainer.test(dataloaders=kwargs['test_loader'],ckpt_path="best", verbose=False)
        

        print('------------------------------------------------')
        #print('Metrics for Checkpoint model:',trainer.callback_metrics,'\n')

        metrics=model.metrics
        best_val_loss_index=pd.Series(metrics['val_losses']).idxmin()
        test_labels=[]
        for i in metrics['test_labels'][0]:
            test_labels.extend(i)
        test_preds=[]
        for i in metrics['test_preds'][0]:
            test_preds.extend(i)

        # train and val loss plots , Till last and best-val loss epochs
        epochs=len(metrics['train_losses'])
        
        print('Total epochs :',epochs)
        plt.plot(range(epochs), metrics['train_losses'], 'g', label='Training loss')
        plt.plot(range(epochs), metrics['val_losses'], 'b', label='validation loss')
        plt.title(f'Till last (Earlystopped or Max) epoch, : {epochs} ')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        print('Best model epoch number :',best_val_loss_index)
        plt.plot(range(best_val_loss_index+1), metrics['train_losses'][:best_val_loss_index+1], 'g', label='Training loss')
        plt.plot(range(best_val_loss_index+1), metrics['val_losses'][:best_val_loss_index+1], 'b', label='validation loss')
        plt.title(f'Till the best {best_loss} epoch : {best_val_loss_index} ')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    #------------ROC-SCORE
    print(f"-----Test-ROC_AUC SCORE (BEST {best_loss} EPOCH)------------:")
    print(roc_auc_score(test_labels,test_preds))
    #--------------------------------------------------
    # computing AAS for each group  over all Encoders  using different criteria
    
    if kwargs['model_type']=='Transformers':
        critiera={'min':min,'max':max,'median':median,'mean':mean}
        critiera={'mean':mean}
        criteria=kwargs['uncertainty_criteria']
        #fig = plt.figure(figsize=(20,17))

        #AAS_avg=[]
        #print(f'\n---- AAS averaged over all groups------------\n')
        #print(model.GAD_scores)
        #print(test_labels)
        #print(test_preds)

        
        #print(sys.exit(0))

        binary_preds=[1 if y_pred > 0.8 else 0 for y_pred in test_preds]
        #print(binary_preds)
        print('RECALL')
        pre,recall,thres=precision_recall_curve(test_labels, test_preds)

        print(auc(recall,pre))
        #print()


        for ind,criterion in enumerate(criteria,1):
            
            AAS_for_each_group=dict(map(lambda x:(x[0],critiera[criterion](x[1])),model.GAD_scores.items()))



            #print(AAS_for_each_group)

            plot_scores=[]
            plot_means=[]

            plt.figure(figsize=(7,5)) 

            for ind,item in enumerate(zip(model.GAD_scores,test_preds,test_labels)):

                if item[2]==1:
                        plt.plot(range(len(model.GAD_scores[item[0]])),model.GAD_scores[item[0]],'r',label='abnormal Traj.-BAS')
                else:
                        plt.plot(range(len(model.GAD_scores[item[0]])),model.GAD_scores[item[0]],'0.8',label='normal Traj.-BAS')
                plot_means.append(AAS_for_each_group[item[0]])
                plot_scores.append(item[1])


            #print(plot_means-plot_scores)

            diff=np.subtract(np.array(plot_means),np.array(plot_scores))
            print('Average (AAS-GAD-Scores: ')
            #print(diff)
            import statistics
            print(statistics.mean(diff))
            plt.ylabel("Block-Attention Score")
            plt.xticks(ticks=[0,1,2,3,4,5])
            plt.xlabel("Encoder-Block")
            plt.title('BAS over all encoder blocks for each trajectory')
            #plt.legend(bbox_to_anchor=(1.1, 1.1))  
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1, 1))      
            plt.show()
            plt.close()


            
            #bins = np.linspace(-10, 10, 30)
            #print(plot_scores,plot_means)
            #X = [143,2,5,6]
            #X=sorted(X)
            X=list(range(len(test_labels)))
            X_axis = np.arange(len(X))
            plt.figure(figsize=(7,5)) 
            plt.bar(X_axis - 0.2, plot_scores, 0.4, label = f'OutputNet-GAD-Score')
            plt.bar(X_axis + 0.2, plot_means, 0.4, label = f'Encoder-Attention-Anomaly-Score')
            #plt.xticks(X_axis, X)
            plt.xticks([])
            plt.xlabel("Groups")
            plt.ylabel("Score")
            plt.title("GAD-Score vs AAS")
            #plt.legend(bbox_to_anchor=(1.1, 1.1))
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.0,1.0)) 
            plt.show()
            plt.close()


           

            
            '''AAS=mean(AAS_for_each_group.values())
            AAS_avg.append(AAS)
            print(f'\n----  Criterion: {criterion} : {AAS}')
            ax = fig.add_subplot(3,3,ind)
            ax.plot(range(len(AAS_for_each_group.values())), AAS_for_each_group.values(), 'g')
            ax.title.set_text(f'Uncertainty with Criterion :{criterion},for each Group')
            ax.set_xlabel('Group')
            ax.set_ylabel('Uncertainty')'''
        '''ax2=fig.add_subplot(3,3,5)
        ax2.title.set_text(f'GAD Scores of each Group') 
        ax2.set_xlabel('Group')
        ax2.set_ylabel('GAD-Score')
        ax2.plot(range(len(test_preds)),test_preds, 'y', label='GAD_scores')
        plt.legend()
        plt.show()'''
    
   
    return roc_auc_score(test_labels,test_preds),metrics['train_losses'][:best_val_loss_index+1],metrics['val_losses'][:best_val_loss_index+1]
    
