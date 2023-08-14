from all_imports import *
from encoder import TransformerEncoder
from position_encoding import PositionalEncoding
from lr_scheduling import CosineWarmupScheduler
from statistics import mean
from sklearn.metrics import roc_auc_score
class TransformerPredictor(pl.LightningModule):
    def __init__(
        self,
        pretrained_filename,
        model_dim,
        input_dim,
        num_classes,
        num_heads,
        num_layers,
        lr,
        optim,
        weight_decay,
        warmup,
        max_epochs,
        max_iters,
        batch_size,
        len_tr,
        len_va,
        len_ts,
        dropout=0.0,
        file_tr='',
        file_va='',
        file_ts='',
        metrics={},
        entity_feat_vec_len=2,
        entity_feat_vec_embed_len=1,
        random_seed=40,
        uncertainty_criteria=['mean','median','min','max'],
        model_type='Transformer'
    ):
        

        self.max_iters=max_iters
        self.metrics=metrics
        self.train_counter=0
        self.test_counter=0
        self.val_counter=0
        self.train_batch_values=[]
        self.train_pred_batch_values=[]
        self.train_lab_batch_values=[]
        self.val_batch_values=[]
        self.val_pred_batch_values=[]
        self.val_lab_batch_values=[]
        self.val_flag=False
        self.test_batch_values=[]
        self.test_pred_batch_values=[]
        self.test_lab_batch_values=[]
        self.len_tr=len_tr
        self.len_va=len_va
        self.len_ts=len_ts
        self.batch_size=batch_size
        self.test_labs=[]
        self.test_preds=[]
        self.test_count=0
        self.GAD_scores={}


        super().__init__()
        self.save_hyperparameters()
        self._create_model()
        
        

    def _create_model(self):
        
        # embedding_block


        self.embedding_block1= nn.Sequential(
            nn.Linear(self.hparams.entity_feat_vec_len, self.hparams.model_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.Sigmoid()
        )
        



        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim,max_len=self.hparams.input_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_feedforward=4 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout
        )
        # Output classifier per sequence element

        self.linear_1=nn.Linear(self.hparams.model_dim, self.hparams.entity_feat_vec_embed_len)
        self.leakyrelu=nn.LeakyReLU()
        self.linear_2=nn.Linear(self.hparams.input_dim, self.hparams.input_dim)
        self.gelu=nn.GELU()
        self.dropout_1=nn.Dropout(self.hparams.dropout)
        self.linear_3=nn.Linear(self.hparams.input_dim, self.hparams.num_classes)
        self.sigmoid=nn.Sigmoid()
        

        

    def forward(self, x,mode, batch_idx,mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            

        """
        #print(x,type(x),batch_idx, mode)
        #print('abve')
        
        x=x.float()
        #print('\nOriginal Batch---:\n')
        #print(x,x.shape)
        shape0=x.shape[0]
        shape1=x.shape[1]
        #print()
        x=torch.reshape(x,(x.shape[1]*x.shape[0],x.shape[2]))
        #print('\nReshaped Batch before inputting to Embedding block---:\n')
        #print(x,x.shape)
        #sys.exit(0)
        x=self.embedding_block1(x)
        #print('\nAfter Embedding block---:\n')
        #print(x,x.shape)
        #sys.exit(0)
        x=torch.reshape(x,(x.shape[0]//shape1,x.shape[0]//shape0,x.shape[1]))
        #print('\nReshapeing the output of Embedding block---:\n')
        #print(x.shape)
        #sys.exit(0)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        #print('\nAfter Positional Encoding block---:\n')
        #print(x.shape)
        
        
        x,attn,GAD_scores = self.transformer(x, mask=mask,batch_idx=batch_idx,mode=mode,GAD_scores=self.GAD_scores)
        
        
       
        


        
        
        self.GAD_scores=GAD_scores
        
        #print('\nAfter Transformer Encoder Stack ---:\n')
        #print(x.shape)
        
        #print('GAD_score in main outside',self.GAD_scores,len(self.GAD_scores))
        #sys.exit(0)
        shape2=x.shape[0]
        shape3=x.shape[1]
        x=torch.reshape(x,(x.shape[1]*x.shape[0],x.shape[2]))
        #print('\nAfter Reshaping the Transformer Encoder Stack ouptut ---:\n')
        #print(x.shape)
        

        ##output block 

        
        x=self.leakyrelu(self.linear_1(x))
        #print('output-block:')
        #print(x.shape)
        x=torch.reshape(x,(x.shape[0]//shape3,x.shape[0]//shape2))
        #print('\nAfter reshaping Embedding block 2 ---:\n')
        #print(x.shape)
        x=self.sigmoid(self.linear_3(self.dropout_1(self.gelu(self.linear_2(x)))))

        #print('\nAfter output_net---:\n')
        #print(x.shape)
        #print(x)
        #sys.exit(0)
        
        return x,attn

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        
        optimizer = self.hparams.optim(self.parameters(), lr=self.hparams.lr,weight_decay=self.hparams.weight_decay)
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters

        
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
    def predict_step(self, batch, batch_idx):
        raise NotImplementedError   
    


class TrajectoryPredictor(TransformerPredictor):
    def _calculate_loss(self, batch,batch_idx, mode="train"):
        
        inp_data, labels = batch
        preds,attn = self.forward(inp_data,mode,batch_idx, add_positional_encoding=True)

        
        loss = F.binary_cross_entropy(preds.view(-1), labels.view(-1).float())
        

        # storing losses and predictions for each Epoch
        if mode=='train':
            self.train_batch_values.append(float(loss.detach()))
            self.train_pred_batch_values.append(list(preds.detach().cpu().numpy().flatten()))
            self.train_lab_batch_values.append(list(labels.detach().cpu().numpy().flatten()))
            self.train_counter+=inp_data.shape[0]
           
            if  self.train_counter==self.len_tr:
                

                self.metrics[mode+'_losses'].append(mean(self.train_batch_values))
                self.metrics[mode+'_preds'].append(self.train_pred_batch_values)
                self.metrics[mode+'_labels'].append(self.train_lab_batch_values)
                self.log("%s_loss" % mode, mean(self.train_batch_values))
                #print(self.losses[mode])
                self.train_counter=0
                self.train_batch_values=[]
                self.train_pred_batch_values=[]
                self.train_lab_batch_values=[]
        elif mode=='val':
            

            if self.val_flag==False:  # unexpected val epoch is running at the start before training epoch
                self.val_counter+=inp_data.shape[0]
                if self.val_counter==self.len_va:
                    self.val_flag=True
                    self.val_counter=0
                #print('Sf1',self.val_counter)
                return None,None
            self.val_batch_values.append(float(loss.detach()))
            self.val_pred_batch_values.append(list(preds.detach().cpu().numpy().flatten()))
            self.val_lab_batch_values.append(list(labels.detach().cpu().numpy().flatten()))
            #print(self.val_batch_values,len(self.val_batch_values))
            self.val_counter+=inp_data.shape[0]
            
            if  self.val_counter==self.len_va:

                self.metrics[mode+'_losses'].append(mean(self.val_batch_values))
                self.metrics[mode+'_preds'].append(self.val_pred_batch_values)
                self.metrics[mode+'_labels'].append(self.val_lab_batch_values)
                self.log("%s_loss" % mode, mean(self.val_batch_values))

                #print(self.losses[mode])
                self.val_counter=0
                self.val_batch_values=[]
                self.val_pred_batch_values=[]
                self.val_lab_batch_values=[]


        else:
            
            self.test_batch_values.append(float(loss.detach()))
            self.test_pred_batch_values.append(list(preds.detach().cpu().numpy().flatten()))
            self.test_lab_batch_values.append(list(labels.detach().cpu().numpy().flatten()))
            self.test_counter+=inp_data.shape[0]
            
                
            if  self.test_counter==self.len_ts:

                    self.metrics[mode+'_losses'].append(mean(self.test_batch_values))
                    self.metrics[mode+'_preds'].append(self.test_pred_batch_values)
                    self.metrics[mode+'_labels'].append(self.test_lab_batch_values)

                    test_labels=[]
                    for i in self.test_lab_batch_values:
                        test_labels.extend(i)
                    test_preds=[]
                    for i in self.test_pred_batch_values:
                        test_preds.extend(i)
                    #print('äää',test_labels,test_preds)
                    self.log("%s_loss" % mode, mean(self.test_batch_values))
                    self.log("%s_roc_auc" % mode, roc_auc_score(test_labels,test_preds))
                    self.test_counter=0
                    self.test_batch_values=[]
                    self.test_pred_batch_values=[]
                    self.test_lab_batch_values=[]





        
        #print(f'epoch: {self.current_epoch}, mode: {mode}, batch {batch_idx}, loss: {loss} \n')
    


        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch,batch_idx, mode="train",)
        
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch,batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch,batch_idx, mode="test")

    def predict_step(self, batch, batch_idx,mode='test'):
        
        return self(batch[0],mode,batch_idx)