import numpy as np
from statistics import median,mean
import sys
def compute_GAD_scores(matrix,record,batch_idx):

    #print('batch_number:',batch_idx)
    '''if batch_idx==0:
        n=8
    elif batch_idx==1:
        n=3
    else:
        n=1

    #n=3'''
    n=int(0.05*matrix.shape[0])  # 5% of the total Groups in a batch which represent anomlousness in the batch
    #print(n)
    #n=5
    if n<1:
        n=1
    
    matrix=matrix.detach().cpu().numpy()
    #print(matrix.shape,matrix)
    #sys.exit(0)
    
    mean_vector=np.mean(matrix,axis=0)
    #print('global_mean_for uncertainity--------')
    distances={}
    #GAD={}
    for i,vector in enumerate(matrix,1):
        distances[i]=[np.linalg.norm(mean_vector-vector),vector]
    # distances contain  [the mean of of all groups(in the batch) represented by 100 vectors of 512 dim- the group, the group]
    #print(distances)
    
    count=0
    avg_topN_b=None
  
    for pair in sorted(distances.items(),key=lambda item: item[1][0],reverse=True): # top n decreasing values
        #print('pari,pari.shape')
        #print(pair,pair[1][1].shape)
        if count<n:
            #print('index,vector value\n')
            #print(pair[0],pair[1])
            #print(np.linalg.norm(mean_vector- pair[1][1]))
            

            if count==0:
                topN_b=pair[1][1]

            else:
                topN_b=np.vstack((topN_b,pair[1][1]))
                #topN_b=np.concatenate((topN_b,pair[1][1]))
            count+=1
        else:
            #print('top n chosen, break!!!!')
            break
    #avg_topN_b=
    #p_avg_topN_b=

    #print('topN grops having maximum distance from the mean group(ie. the  most anomalous groups in terms of distance')
    #'topN grops having maximum distance from the mean group(ie. the  most anomalous groups in terms of distance'
    #print(topN_b,topN_b.shape)
    topN_b=np.reshape(topN_b,(n,mean_vector.shape[0],mean_vector.shape[1]))
    #print(topN_b,topN_b.shape)


    # calculate the average(or median)-distance avgTopN_b of those top n high-distance-sample-matrices from the mean for each block
    p_avg_topN_b=np.mean(topN_b,axis=0)
    #print('p_avg_topN_b\n')
    #print(p_avg_topN_b)
    #sys.exit(0)


    #-----calculating uncertainit for each group and then averaging
    GAD_scores_all=0
    GAD_scores=[]
    for i,vector in enumerate(matrix,1):

        distances[i]=[np.linalg.norm(mean_vector-vector),vector]
        

        
        
        
        
        GAD_score=min(1,distances[i][0]/np.linalg.norm(mean_vector-p_avg_topN_b))
        #print('Gad_score',GAD_score)
        GAD_scores_all+=GAD_score

        try:
            record[f'GAD_score_batch_{batch_idx}_Group_{i}'].append(GAD_score)                       
        except:  
            record[f'GAD_score_batch_{batch_idx}_Group_{i}']=[]                                    # storing GAD score for instance of each encoder 
            record[f'GAD_score_batch_{batch_idx}_Group_{i}'].append(GAD_score)

        
        GAD_scores.append(GAD_score)
         
        
    

    #print('-----', record)

    return record