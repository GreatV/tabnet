from pytorch_tabnet.tab_network import TabNet
import torch


if __name__ == "__main__":
    cat_dims = [73, 9, 16, 16, 7, 15, 6, 5, 2, 119, 92, 94, 42]
    cat_idxs = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    cat_emb_dim = [2] * 13
    group_attention_matrix = torch.rand(10, 14)
    model = TabNet(14,
                   2,
                   cat_dims=cat_dims,
                   cat_idxs=cat_idxs,
                   cat_emb_dim=cat_emb_dim,
                   group_attention_matrix=group_attention_matrix)
    x = torch.rand(1024, 14)
    try:
        torch.export.export(model, (x, ))
        print ("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print ("[JIT] torch.export failed.")
        raise e
