# Author: Lu Tongyu
# Please modify the items for yourself
# date: 20200714

class configurations(object):
    def __init__(self):
        self.batch_size = 1
        self.emb_dim = 64 # make sure that emb_dim/num_heads is an integer!
        self.splitted_dim_list = [16,2,46]
        self.class_dim = self.splitted_dim_list[0]+self.splitted_dim_list[1]

        self.class_contents_int2class = {0:'controller',1:'4/4',2:'2/2',3:'3/4',4:'6/8',5:'5/4',6:'3/2',7:'2/4',8:'9/8',9:'12/8',10:'12/16',11:'6/4',12:'others'}
        self.class_contents_class2int = {value:key for key, value in self.class_contents_int2class.items()}
        self.grouping_int2class = {0:'controller',1:'binary',2:'ternary',3:'quintet',4:'others'}
        self.grouping_class2int = {value:key for key, value in self.grouping_int2class.items()}
        self.binary_list = ['4/4','2/2','6/8','2/4','12/8','12/16']
        self.ternary_list = ['3/4','3/2','9/8','6/4']
        self.quintet_list = ['5/4','5/2','5/8']
        
        self.classifier_out_len_list = [len(self.class_contents_int2class),len(self.grouping_int2class)]
        self.n_layers = 6 #6
        self.num_heads = 8 #8
        self.dropout = 0.5
        self.ffn_dim = 1024
        self.learning_rate = 0.0001
        self.max_output_len = 800
        self.num_steps = 20000
        self.store_steps = 4000
        self.summary_steps = 1000
        self.mask_prob = 0.2 #0.5
        self.head_len = 10
        self.bias_tokens_n = 20
        self.vq_vocab_size_factor = 5
        self.sampling = True
        self.load_model = False
        self.store_model_path = "/" #Please change the path for your environment
        self.load_model_path = "/model_"+str(self.num_steps) #Please change the path for your environment
        self.data_path = "/data_folder/" #Please change the path for your environment

config = configurations()
