from overrides import overrides

from bonfire.main.data.benchmark.musk.musk_dataset import MuskDataset
from bonfire.main.model import mil_aggregators as agg
from bonfire.main.model import modules as mod


def get_model_clzs():
    return [MuskInstanceSpaceNN, MuskEmbeddingSpaceNN, MuskAttentionNN, MuskGNN, MuskMiLstm]


class MuskInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device, d_enc=512, ds_enc_hid=(), ds_agg_hid=(256, 64), dropout=0.45, agg_func_name='mean'):
        encoder = mod.FullyConnectedStack(MuskDataset.d_in, ds_enc_hid, d_enc,
                                          final_activation_func=None, dropout=dropout)
        aggregator = agg.InstanceAggregator(d_enc, ds_agg_hid, MuskDataset.n_classes, dropout, agg_func_name)
        super().__init__(device, MuskDataset.n_classes, MuskDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-5,
        }


class MuskEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device, d_enc=256, ds_enc_hid=(128,), ds_agg_hid=(), dropout=0.25, agg_func_name='max'):
        encoder = mod.FullyConnectedStack(MuskDataset.d_in, ds_enc_hid, d_enc,
                                          final_activation_func=None, dropout=dropout)
        aggregator = agg.EmbeddingAggregator(d_enc, ds_agg_hid, MuskDataset.n_classes, dropout, agg_func_name)
        super().__init__(device, MuskDataset.n_classes, MuskDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-3,
        }


class MuskAttentionNN(models.AttentionNN):

    def __init__(self, device, d_enc=128, ds_enc_hid=(128, 256), ds_agg_hid=(), dropout=0.15, d_attn=256):
        encoder = mod.FullyConnectedStack(MuskDataset.d_in, ds_enc_hid, d_enc,
                                          final_activation_func=None, dropout=dropout)
        aggregator = agg.MultiHeadAttentionAggregator(1, d_enc, ds_agg_hid, d_attn, MuskDataset.n_classes, dropout)
        super().__init__(device, MuskDataset.n_classes, MuskDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-5,
        }


class MuskGNN(models.ClusterGNN):

    def __init__(self, device,
                 d_enc=128, ds_enc_hid=(), d_gnn=64, ds_gnn_hid=(128, 256), ds_fc_hid=(128,), dropout=0.2):
        encoder = mod.FullyConnectedStack(MuskDataset.d_in, ds_enc_hid, d_enc,
                                          final_activation_func=None, dropout=dropout)
        super().__init__(device, MuskDataset.n_classes, MuskDataset.n_expected_dims, encoder,
                         d_enc, d_gnn, ds_gnn_hid, ds_fc_hid, dropout)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-5,
        }


class MuskMiLstm(models.MiLstm):

    def __init__(self, device, d_enc=128, ds_enc_hid=(256,), d_lstm_hid=128, n_lstm_layers=2,
                 bidirectional=True, ds_fc_hid=(64,), dropout=0.2):
        encoder = mod.FullyConnectedStack(MuskDataset.d_in, ds_enc_hid, d_enc,
                                          final_activation_func=None, dropout=dropout)
        aggregator = agg.LstmEmbeddingSpaceAggregator(d_enc, d_lstm_hid, n_lstm_layers, bidirectional,
                                                      dropout, ds_fc_hid, MuskDataset.n_classes)
        super().__init__(device, MuskDataset.n_classes, MuskDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-5,
        }
