import torch
import esm
import numpy as np

def esm_feature(protein, pro_seq):
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    model.eval()  
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [(protein, pro_seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations = token_representations[0, 1: len(pro_seq) - 1].mean(0)

    return sequence_representations


def save_representation_to_file(representation, file_path):
    representation_np = representation.cpu().numpy()
    with open(file_path, 'wb') as f:
        np.save(f, representation_np)

