from all_imports import *
from lr_scheduling import CosineWarmupScheduler
from statistics import mean
from sklearn.metrics import roc_auc_score
class GRUNet(pl.LightningModule):
    def __init__(
        self,
        pretrained_filename,
        model_dim,
        input_dim,
        num_classes,
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
        n_layers = 2,
        random_seed=40,
        model_type='Baseline'
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
        


        

        self.gru = nn.GRU(self.hparams.model_dim, self.hparams.model_dim, self.hparams.n_layers, batch_first=True, dropout=self.hparams.dropout)
        


        self.linear_1=nn.Linear(self.hparams.model_dim, self.hparams.entity_feat_vec_embed_len)
        self.leakyrelu=nn.LeakyReLU()
        self.linear_2=nn.Linear(self.hparams.input_dim, self.hparams.input_dim)
        self.gelu=nn.GELU()
        self.dropout_1=nn.Dropout(self.hparams.dropout)
        self.linear_3=nn.Linear(self.hparams.input_dim, self.hparams.num_classes)
        self.sigmoid=nn.Sigmoid()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden1 = weight.new(self.hparams.n_layers, batch_size, self.hparams.model_dim).zero_().to(device)
        #hidden=torch.zeros(self.hparams.n_layers, batch_size, self.hparams.model_dim) 
        return hidden1

        

    def forward(self, x, h):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            

        """
        out, h = self.gru(x, h)
        shape2=out.shape[0]
        shape3=out.shape[1]
        out=torch.reshape(out,(out.shape[1]*out.shape[0],out.shape[2]))


        ####### output block ######

        out=self.leakyrelu(self.linear_1(out))
        out=torch.reshape(out,(out.shape[0]//shape3,out.shape[0]//shape2))
        out=self.sigmoid(self.linear_3(self.dropout_1(self.gelu(self.linear_2(out)))))
        ######         ######
    
        return out,h

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


class GRU_RNN(GRUNet):
    def _calculate_loss(self, batch,batch_idx, mode="train"):
        
        inp_data, labels = batch
        inp_data=inp_data.float()
        
        shape0=inp_data.shape[0]
        shape1=inp_data.shape[1]
        inp_data=torch.reshape(inp_data,(inp_data.shape[1]*inp_data.shape[0],inp_data.shape[2]))
        
        inp_data=self.embedding_block1(inp_data)
        
        inp_data=torch.reshape(inp_data,(inp_data.shape[0]//shape1,inp_data.shape[0]//shape0,inp_data.shape[1]))
        h = self.init_hidden(inp_data.shape[0])
        
        preds,h = self.forward(inp_data,h)

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
