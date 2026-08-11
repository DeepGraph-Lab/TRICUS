from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # 基础训练参数
        parser.add_argument('--loss_freq', type=int, default=400,
                            help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count')
        parser.add_argument('--last_epoch', type=int, default=-1,
                            help='starting epoch count for scheduler intialization')
        parser.add_argument('--niter', type=int, default=200,
                            help='total epoches')
        parser.add_argument('--lr', type=float, default=1e-5,
                            help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, default=0.0,
                            help='loss weight for l2 reg')
        parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                            help='Optimizer (default: "adamw"')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
        parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                            help='LR scheduler (default: "cosine"')

        # GNN参数
        parser.add_argument('--dropout', type=float, default=0.,
                            help='dropout probability')
        # GCN
        parser.add_argument('--gcn_in_dim', type=int, default=768,
                            help='also for linear_out_size')
        parser.add_argument('--gcn_out_dim', type=int, default=768,
                            help='also for gat_in_dim')
        parser.add_argument('--gcn_layers', type=int, default=2,
                            help='layer num for GCN')

        parser.add_argument('--pred_hid_size', type=int, default=1536,
                            help='predictor hidden size')

        # GAT
        parser.add_argument('--gat_dropout', type=float, default=0.1,
                            help='gat dropout probability')
        parser.add_argument('--gat_layers', type=int, default=2,
                            help='layer num for GAT')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='head num for GAT')
        parser.add_argument('--gat_hid_dim', type=int, default=768,
                            help='also for gat_out_dim')

        # ==================== GMF参数 ====================
        parser.add_argument('--gmf_dim', type=int, default=128,
                            help='GMF embedding dimension')
        parser.add_argument('--gmf_reg', type=float, default=0.01,
                            help='L2 regularization for GMF')
        parser.add_argument('--gmf_num_prototypes', type=int, default=64,
                            help='number of shared prototypes in GMF')

        # ==================== 损失权重 ====================
        parser.add_argument('--task_weight', type=float, default=1.0,
                            help='weight for GNN BCE loss')
        parser.add_argument('--gmf_recon_weight', type=float, default=0.5,
                            help='weight for GMF reconstruction loss')

        # ==================== 消融 ====================
        parser.add_argument('--gcn_module', type=int, default=1,
                            help='1 for using gcn module,0 for not using gcn module')
        parser.add_argument('--gat_module', type=int, default=1,
                            help='1 for using gat module,0 for not using gat module')
        parser.add_argument('--gmf_module', type=int, default=1,
                            help='1 for using gmf module,0 for not using gmf module')
        parser.add_argument('--random_emb', type=int, default=0,
                            help='0 for using pre-trained emb,1 for using random emb')
        parser.add_argument('--finetune_module', type=int, default=1,
                            help='1 for using fine-tuning module,0 for not using fine-tuning module')

        self.isTrain = True
        return parser