import copy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from sentence_transformers import SentenceTransformer

import numpy as np 
def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class SBert(nn.Module):
    """Some Information about SBert"""
    def __init__(self, device):
        super(SBert, self).__init__()
        model1 = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        self.model = nn.DataParallel(model1, device_ids=[0,1,2], dim=1)
        self.device = device
        self.to(device)
        #self.model = nn.DataParallel(model)
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        mask = []
        # res_msk = []
        # res_labels =[]

        # def add_to_list_mask(res_mask, element):
        #     res_mask.append(element)
            
        for d in data:
            temp = []
            # temp_msk = []
            # temp_labels = []
            for i in range(width - len(d)):
                temp.append(np.zeros(768))
                #temp_msk.append(0)
                #temp_labels.append(0)
            
#            print("TAILLE DE D", len(d))
            # print("tempmask2 :", [1]*len(d) + [0]*(width - len(d)))
            # temp_mask = [1]*len(d)  + [0]*(width - len(d))



            #msk = [1 for _ in range(len(d))]
            #new_labels = [1 for _ in range(len(d))]

            for j in temp:
                d.append(j)
            #msk_batch = msk + temp_msk
            #labels_batch = new_labels + temp_labels
            # res_msk.append(msk_batch) 
            #res_msk.append([1]*len(d)  + [0]*(width - len(d)))
            #add_to_list_mask(res_msk, [1]*len(d)  + [0]*(width - len(d)))
            #res_labels.append(labels_batch)

        
        return data#, res_msk#, res_labels


    # def _pad2(self, data, pad_id, width=-1):
    #     if (width == -1):
    #         width = max(len(d) for d in data)
    #     #print("GRAND DATA size:", data.size())
    #     for d in data:
    #         print('LEN DS DATA', len(d))
    #     rtn_data = [[1]*len(d)  + [0] * (width - len(d)) for d in data]
    #     print("rtn_data",rtn_data)
    #     return rtn_data

    def forward(self, x):
        """
        args:
        x : sentences batched 
        #x = [["phrase 1", "phrase 2"], ["phrase 3", "phrase 4"]]

        res = [[np.array(), np.array()], [np.array(), np.array()]]

        
        """
#        for i in x:
 #           print("len src txt", len(i))
  #      print("src_txt :", x)
        #padded_txt = self._pad(x, 0)
        pre_res = []

        #print("padded_txt :", padded_txt)
        for article in x:
            pre_res.append(self.model.module.encode(article, show_progress_bar=False))
  
        #print("dim de res avant padding", len(pre_res))
        # res, mask, labels_new= self._pad(pre_res, 0)[0], self._pad(pre_res,0)[1], self._pad(pre_res, 0)[1]
        res = self._pad(pre_res,0)
        #mask = self._pad(pre_res,0)[1]

        #print("res :", res)

       # print("mask avant tensor", mask )
   #     print("len(res) : ", len(res))
    #    print("len(res[0])", len(res[0]))
     #   print("len(res[0][0])", len(res[0][0]))

        t_res = torch.tensor(res)
        t_res = t_res.type(torch.FloatTensor)

        t_res = t_res.to(self.device)
        #t_res = torch.FloatTensor(res)
        #mask = torch.tensor(mask)
        #labels = torch.tensor(labels_new)
        return t_res#, mask#, labels
        


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        print("DEVICE :", device)
        self.device = device
        #self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.SBert = SBert(device)
        self.SBert.to(device)
        self.ext_layer = ExtTransformerEncoder(768, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls, src_txt):
        # top_vec = self.bert(src, segs, mask_src)
        # sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        # sents_vec = sents_vec * mask_cls[:, :, None].float()
        # sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)

        ###########################

      #  print("src_txt :", src_txt)
       # print("mask_cls", mask_cls)
        top_vec = self.SBert(src_txt)
        top_vec = top_vec.to(self.device)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
        ###########################
        
        # print("size msk cls initial", mask_cls.size())
        

        # #top_vec, mask= self.SBert(src_txt)
        # top_vec, mask= self.SBert(src_txt)
        # print("size top_vec", sents_vec.size())

        # #Since we take eveything, supposed to be of the same size as top_vec [5,65,768] for ex
        # sents_vec = top_vec[torch.arange(top_vec.size(0).unsqueeze(1), :)]

        # #Should be the same as size sents_vec
        # print("Size mask after SBert :", mask.size())
        # print("Mask after SBert :", mask)


        # print("Applying the mask to the sentences ")
        # sents_vec = sents_vec * mask[:, :, None].float()

        # #Should be [5,65]
        # print("Size sents_vec after the mask")

        # sent_scores = self.ext_layer(vec, mask_cls).squeeze(-1)
        # print("size sent_scores apres squeeze apres ExtSummarize",sent_scores.size())
        # return sent_scores, mask, labels_new
        # #return sent_scores, mask_cls


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
       # self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.SBert = SBert(device)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        #top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None



       # top_vec = self.SBert(src_txt)
       # top_vec = top_vec.to(device)
       # dev_st
       # top_vec = self.SBert(src_txt)
       # top_vec = top_vec.to(self.device)
       # sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
       # sents_vec = sents_vec * mask_cls[:, :, None].float()
       # sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
       # return sent_scores, mask_cls


