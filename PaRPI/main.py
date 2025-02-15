import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from utils.PaRPI import PaRPI
from utils.gen_bert_embedding import circRNABert, save_representation_to_file
import torch.utils.data
from transformers import BertModel, BertTokenizer
from utils.train_loop import train, validate
from utils.utils import read_csv, myDataset, GradualWarmupScheduler, param_num, split_dataset, seq2kmer, count_split_dataset, read_protein_fasta, load_cell_matrix_from_txt
from utils import esm
from utils.rna_graph import *
import glob
from sklearn.preprocessing import MinMaxScaler

def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    # print("[Info] cudnn.deterministic set to True. CUDNN-optimized code may be slow.")
    
def collate_fn(batch):
    graphs, berts, structs, proteins, labels = zip(*batch)

    batched_graph = dgl.batch(graphs)  

    berts = np.array(berts)
    berts = torch.tensor(berts, dtype=torch.float32)
    structs = np.array(structs)
    structs = torch.tensor(structs, dtype=torch.float32)
    proteins = np.array(proteins)
    proteins = torch.tensor(proteins, dtype=torch.float32)
    labels = np.array(labels)
    labels = torch.tensor(labels, dtype=torch.float32)

    return batched_graph, berts, structs, proteins, labels

def main(args):
    

    
    try:
        from termcolor import cprint
    except ImportError:
        cprint = None

    try:
        from pycrayon import CrayonClient
    except ImportError:
        CrayonClient = None


    def log_print(text, color=None, on_color=None, attrs=None):
        if cprint is not None:
            cprint(text, color=color, on_color=on_color, attrs=attrs)
        else:
            print(text)


    fix_seed(args.seed)  # fix seed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #load BERT model
    bert_model_path = args.BERT_model_path
    tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=False)
    model = BertModel.from_pretrained(bert_model_path)
    model = model.to(device)
    model = model.eval()
    
    max_length = 101

    cell_name = args.cell
    data_path = args.data_path
    data_name = args.data_name
    
    if args.train:
        
        scaler = MinMaxScaler()
        
        train_graph_all, test_graph_all = [], []
        
        lenth = 0
        lenth_test = 0
        lenth_tra = 0
        lenth2_tra = 0
        lenth2_test = 0  
        protein_tra = None
        
        tsv_files = sorted(glob.glob(os.path.join(data_path, '*_{}.tsv').format(cell_name)))    
        
        for i in tsv_files:
            _, _, _, label_1 = read_csv(i)
            [train_label], [test_label] = count_split_dataset(label_1)  
            num_rows = len(label_1)
            lenth = lenth + num_rows
            lenth_tra += len(train_label)
            lenth_test += len(test_label)
        
        filename1 = 'dataset/bat/bert_embedding_tra.dat'
        filename2 = 'dataset/bat/bert_embedding_test.dat'
        shape1 = (lenth_tra, 768, 99)
        shape2 = (lenth_test, 768, 99)
        bert_embedding_tra = np.memmap(filename1, dtype=np.float32, mode='w+', shape=shape1)
        bert_embedding_test = np.memmap(filename2, dtype=np.float32, mode='w+', shape=shape2)

        filename3 = 'dataset/bat/pro_tra.dat'
        filename4 = 'dataset/bat/pro_test.dat'
        shape3 = (lenth_tra, 1280, 1)
        shape4 = (lenth_test, 1280, 1)
        protein_tra = np.memmap(filename3, dtype=np.float32, mode='w+', shape=shape3)
        protein_test = np.memmap(filename4, dtype=np.float32, mode='w+', shape=shape4)

        filename5 = 'dataset/bat/structure_tra.dat'
        filename6 = 'dataset/bat/structure_test.dat'
        shape5 = (lenth_tra, 1, 101)
        shape6 = (lenth_test, 1, 101)
        structure_tra = np.memmap(filename5, dtype=np.float32, mode='w+', shape=shape5)
        structure_test = np.memmap(filename6, dtype=np.float32, mode='w+', shape=shape6)

        filename9 = 'dataset/bat/label_tra.dat'
        filename10 = 'dataset/bat/label_test.dat'
        shape9 = (lenth_tra, 1)
        shape10 = (lenth_test, 1)
        label_tra = np.memmap(filename9, dtype=np.float32, mode='w+', shape=shape9)
        label_test = np.memmap(filename10, dtype=np.float32, mode='w+', shape=shape10)
        
        for data_name in tsv_files:
            sequences, structs, prot, label = read_csv(data_name)
            
            
            data_name = data_name.rsplit('/', 1)[-1]  
            data_name = data_name.rsplit('.', 1)[0]  
            parts = data_name.split("_")
            
            #protein feature
            pro = prot.iloc[0]
            pro_seq = read_protein_fasta("dataset/protein/{}.fasta".format(pro))
            if os.path.isfile("dataset/esm/{}.npy".format(data_name)):
                pro_seq_feature = np.load("dataset/esm/{}.npy".format(data_name))
            else:
                pro_seq_feature = esm.esm_feature(pro, pro_seq)
                #esm.save_representation_to_file(pro_seq_feature, "dataset/esm/{}.npy".format(data_name))
            pro_seq_feature = pro_seq_feature.reshape((1280, 1))
            protein = np.tile(pro_seq_feature, (len(prot), 1, 1))

            #RNAbert feature
            if os.path.isfile("dataset/bert/{}.npy".format(data_name)):
                bert_embedding = np.load("dataset/bert/{}.npy".format(data_name))
            else:
                bert_embedding = circRNABert(list(sequences), model, tokenizer, device, 3)  
                bert_embedding = bert_embedding.transpose([0, 2, 1])  

            #RNAstructure feature
            structure = np.zeros((len(structs), 1, max_length))  
            for i in range(len(structs)):
                struct = structs[i].split(',')
                ti = [float(t) for t in struct]
                ti = np.array(ti).reshape(1, -1)
                structure[i] = np.concatenate([ti], axis=0)
            
            [train_seq, train_emb, train_struc, train_pro, train_label], [test_seq, test_emb, test_struc, test_pro, test_label] = \
                split_dataset(sequences, bert_embedding, structure, protein, label)  

            data_to_save = {
                                'test_seq': test_seq,
                                'test_emb': test_emb,
                                'test_struc': test_struc,
                                'test_pro': test_pro,
                                'test_label': test_label
                            }
            save_path = os.path.join('./dataset/test_set', '{}.npz'.format(data_name))
            np.savez(save_path, **data_to_save)

            if os.path.exists('./dataset/dgl/{}_train.pt'.format(data_name)) and os.path.exists('./dataset/dgl/{}_test.pt'.format(data_name)):
                print("Loading {} graph data...".format(data_name))
                train_graph = torch.load('./dataset/dgl/{}_train.pt'.format(data_name))
                test_graph = torch.load('./dataset/dgl/{}_test.pt'.format(data_name))

                train_graph_all.extend(train_graph)
                test_graph_all.extend(test_graph)
            else:
                print("Generating {} graph data...".format(data_name))
                train_seq = [seq[1:-1] for seq in train_seq]  
                test_seq = [seq[1:-1] for seq in test_seq]  

                train_graph = convert2dglgraph(train_seq)
                test_graph = convert2dglgraph(test_seq)
                #torch.save(train_graph, './dataset/dgl/{}_train.pt'.format(data_name))
                #torch.save(test_graph, './dataset/dgl/{}_test.pt'.format(data_name))

            train_graph_all.extend(train_graph)
            test_graph_all.extend(test_graph)

            
            lenth1_tra = lenth2_tra
            lenth2_tra = lenth2_tra + len(train_emb)
            lenth1_test = lenth2_test
            lenth2_test = lenth2_test + len(test_emb)
            
            bert_embedding_tra[lenth1_tra:lenth2_tra] = train_emb
            bert_embedding_test[lenth1_test:lenth2_test] = test_emb

            protein_tra[lenth1_tra:lenth2_tra] = train_pro
            protein_test[lenth1_test:lenth2_test] = test_pro

            structure_tra[lenth1_tra:lenth2_tra] = train_struc
            structure_test[lenth1_test:lenth2_test] = test_struc

            label_tra[lenth1_tra:lenth2_tra] = train_label
            label_test[lenth1_test:lenth2_test] = test_label
            
            print("{} over".format(data_name))
            
        train_set = myDataset(train_graph_all, bert_embedding_tra, structure_tra, protein_tra, label_tra)
        test_set = myDataset(test_graph_all, bert_embedding_test, structure_test, protein_test, label_test)
        
        train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_set, collate_fn=collate_fn, batch_size=256*8, shuffle=False)

        model = PaRPI().to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
        optimizer = torch.optim.AdamW(
                                        model.parameters(), 
                                        lr=0.001,
                                        betas=(0.9, 0.999), 
                                        eps=1e-08, 
                                        weight_decay=0.01, 
                                        amsgrad=False, 
                                        maximize=False, 
                                        foreach=None, 
                                        capturable=False,
                                        differentiable=False, 
                                        fused=None
                                    )
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=8, total_epoch=float(200), after_scheduler=None)

        best_auc = 0
        best_acc = 0
        best_epoch = 0

        model_save_path = args.model_save_path

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        early_stopping = args.early_stopping

        param_num(model)

        for epoch in range(1, 200):
            t_met = train(model, device, train_loader, criterion, optimizer, epoch, batch_size=256)
            v_met, _, _ = validate(model, device, test_loader, criterion)
            scheduler.step()        
            
            lr = scheduler.get_lr()[0]
            color_best = 'green'
            if best_auc < v_met.auc:
                best_auc = v_met.auc
                best_acc = v_met.acc
                
                best_tpr = v_met.tp/(v_met.tp + v_met.fn) if (v_met.tp + v_met.fn) >0 else 0
                best_tnr = v_met.tn/(v_met.tn + v_met.fp) if (v_met.tn + v_met.fp) >0 else 0
                best_precision = v_met.tp/(v_met.tp + v_met.fp) if (v_met.tp + v_met.fp) >0 else 0
                best_F1 = 2*(best_precision * best_tpr)/(best_precision + best_tpr)
                best_prc = v_met.prc
                
                best_epoch = epoch
                color_best = 'red'
                path_name = os.path.join(model_save_path, cell_name+'.pth')
                torch.save(model.state_dict(), path_name)
            if epoch - best_epoch > early_stopping:
                print("Early stop at %d, %s " % (epoch, 'PaRPI'))
                break
            line = '{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} lr: {:.6f}'.format(
                cell_name, epoch, t_met.other[0], t_met.acc, t_met.auc, lr)
            log_print(line, color='green', attrs=['bold'])

            line = '{} \t Test  Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} ({:.4f}) {}'.format(
                cell_name, epoch, v_met.other[0], v_met.acc, v_met.auc, best_auc, best_epoch)
            log_print(line, color=color_best, attrs=['bold'])

        print("{} auc: {:.4f} acc: {:.4f}".format(cell_name, best_auc, best_acc))
        
        
    if args.validate:  
               
        path = "./dataset/test_set/"
        npz_files = sorted(glob.glob(os.path.join(path, '*_{}.npz').format(cell_name)))
        
        for file in npz_files:
            data_name = file.rsplit('/', 1)[-1]  
            data_name = data_name.rsplit('.', 1)[0]  
            loaded_data = np.load(file, allow_pickle=True)
            test_graph = torch.load('./dataset/dgl/{}_test.pt'.format(data_name))
            test_emb = loaded_data['test_emb']
            test_struc = loaded_data['test_struc']
            test_pro = loaded_data['test_pro']
            test_label = loaded_data['test_label']

            test_set = myDataset(test_graph, test_emb, test_struc, test_pro, test_label)
            test_loader = DataLoader(test_set, collate_fn=collate_fn, batch_size=256*8, shuffle=False)

            model = PaRPI().to(device)
            model_file = os.path.join(args.model_save_path, cell_name + '.pth')
            if not os.path.exists(model_file):
                print('Model file does not exitsts! Please train first and save the model')
                exit()
            model.load_state_dict(torch.load(model_file))

            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

            met, y_all, p_all = validate(model, device, test_loader, criterion)


            best_auc = met.auc
            best_acc = met.acc
            print("{} auc: {:.4f} acc: {:.4f}".format(data_name, best_auc, best_acc))
                
    if args.prediction_aware:
        
        scaler = MinMaxScaler()
        protein_tra = None   
        
        sequences, structs, prot, label = read_csv(os.path.join(data_path, '{}.tsv'.format(data_name)))
        data_name = data_name.rsplit('/', 1)[-1]  
        data_name = data_name.rsplit('.', 1)[0]  
        parts = data_name.split("_")

        #protein feature
        pro = prot.iloc[0]
        pro_seq = read_protein_fasta("dataset/protein/{}.fasta".format(pro))
        pro_seq_feature = esm.esm_feature(pro, pro_seq)
        pro_seq_feature = pro_seq_feature.reshape((1280, 1))
        protein = np.tile(pro_seq_feature, (len(prot), 1, 1))

        #RNAbert feature
        bert_embedding = circRNABert(list(sequences), model, tokenizer, device, 3)  
        bert_embedding = bert_embedding.transpose([0, 2, 1])  

        #RNAstructure feature
        structure = np.zeros((len(structs), 1, max_length))  
        for i in range(len(structs)):
            struct = structs[i].split(',')
            ti = [float(t) for t in struct]
            ti = np.array(ti).reshape(1, -1)
            structure[i] = np.concatenate([ti], axis=0)

        print("Generating {} graph data...".format(data_name))
        test_seq = [seq[1:-1] for seq in sequences]  
        test_graph = convert2dglgraph(test_seq)
        print("{} over".format(data_name))
            
        test_set = myDataset(test_graph, bert_embedding, structure, protein, label)
        test_loader = DataLoader(test_set, collate_fn=collate_fn, batch_size=256*8, shuffle=False)

        model = PaRPI().to(device)
        model_file = os.path.join(args.model_save_path, cell_name + '.pth')
        if not os.path.exists(model_file):
            print('Model file does not exitsts! Please train first and save the model')
            exit()
        model.load_state_dict(torch.load(model_file))

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        met, y_all, p_all = validate(model, device, test_loader, criterion)


        best_auc = met.auc
        best_acc = met.acc
        print("{} auc: {:.4f} acc: {:.4f}".format(data_name, best_auc, best_acc))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to PaRPI!')
    parser.add_argument('--cell', default='K562', type=str, help='The cell line of RBPs')
    parser.add_argument('--data_name', default='LIN28A_H9', type=str, help='The data name of RBP')
    parser.add_argument('--data_path', default='./dataset/clip_data', type=str, help='The data path')
    parser.add_argument('--BERT_model_path', default='./BERT_Model', type=str, help='BERT model path, in case you have another BERT')
    parser.add_argument('--model_save_path', default='./results/model', type=str, help='Save the trained model for dynamic prediction')

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--validate', default=False, action='store_true')
    parser.add_argument('--prediction_aware', default=False, action='store_true')

    parser.add_argument('--seed', default=1024, type=int, help='The random seed')
    parser.add_argument('--early_stopping', type=int, default=20)
    args = parser.parse_args()
    main(args)

