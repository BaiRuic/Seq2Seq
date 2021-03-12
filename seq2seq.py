import torch 
import torch.nn as nn
import sys
import numpy as np
import random
import utils.my_logging as my_logging

my_log  = my_logging.My_Logging(logname="log.log")
class Encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size:int, hidden_size:int):
        '''
        param:
            input_size:    the number of features in the input X
            hidden_size:   the number of features in the hidden state h
        '''
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=1, 
                            bias=True, 
                            batch_first=True,)
    def forward(self, inputs):
        '''
        params:
            inputs  shape:[batch_size, seq_len, features]
                    function: 作为编码器的输入
        return: 
            hidden  shape:[1, batch_size, hidden_size]
            cell    shape:[1, batch_size, hidden_size]
                    作为编码器的输出，即LSTM的最后一个 状态 和 隐藏态
        '''
        outputs, (hidden, cell) = self.lstm(inputs)
        return hidden, cell

class Decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, dropout=0.2):
        '''
        param:
            input_size:  the number of features in the input X
            hidden_size: the number of features in the hidden state h, is similar to hidden_size of Encoder
        '''
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True)

        self.fc = nn.Linear(in_features=self.hidden_size, 
                            out_features=self.output_size, 
                            bias=True)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, prev_hidden, prev_cell):
        '''
        params:
            inputs      shape: [batch, seq_len=1, input_size=1]
            prev_hidden shape: [num_layers * rnn_directions=1, batch, hidden_size]
            prev_cell   shape: [1, batch,hidden_size]
                        note:  prev_hidden和prev_cell 是编码器的输出状态，所以解码器和编码器的hidden_size应当一样
        return:
            prediction  shape: [batch_size, seq_len=1, output_size=1]
            hidden      shape: [num_layers * rnn_directions=1, batch, hidden_size]
            cell        shape: [num_layers * rnn_directions=1, batch, hidden_size]
        '''
       
        batch_size = inputs.shape[0]
        # outputs [batch, seq_len=1, hidden_size]
        # hidden [1, batch_size, hidden_size] 所有 torch.equal(output.squeeze(1), hidden.squeeze(0) 是True
        outputs, (hidden, cell) = self.lstm(inputs,(prev_hidden, prev_cell))
        
        # prediction [batch, seq_len=1, outputs_size=1]
        prediction = self.fc(outputs.view(batch_size, self.hidden_size)).unsqueeze(dim=1)
        return prediction, hidden, cell 

class RNN_Seq2Seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    def __init__(self, input_size, hidden_size, predict_seqlen):
        '''
        input_size: 输出特征数
        hidden_size: 隐藏层的特征数
        predict_seqlen: 需要预测序列的长度
        '''
        super(RNN_Seq2Seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.predict_seqlen = predict_seqlen

        self.encoder = Encoder(input_size=input_size,hidden_size=self.hidden_size)
        self.decoder = Decoder(input_size=1, hidden_size=self.hidden_size)

    def forward(self, inputs):
        '''
        params:
            inputs [batch_size, seq_len, feature] 输入到编码器的数据
        return:
            outputs:[batch_size, predict_seqlen, 1] # 最后的1表示预测输出的维度为1
        '''
        batch_size = inputs.shape[0]
        
        # 编码
        hidden, cell = self.encoder(inputs)

        # 解码器初始输入：为inputs最后一个时间步的待预测值
        x = inputs[:,-1,0].unsqueeze(dim=1).unsqueeze(dim=2)

        # 存放解码器 的每一个时间步 的输出
        outputs = []

        # Decodeing
        for i in range(self.predict_seqlen): 
            pred, hidden, cell = self.decoder(x,hidden, cell)
            x = pred
            outputs.append(pred)   # pred[batch_size, seq_len=1, 1] 每次存进去一个时间步的，最后对时间轴拼接
        
        # 对时间维度拼接
        outputs = torch.cat(outputs,dim=1)

        return outputs
            

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    setup_seed(20) # 设置随机种子
    batch_size = 6128
    seq_len = 14
    feature = 4
    pred_seq_len = 6
    x = torch.rand((batch_size, seq_len, feature))
    y = torch.rand(batch_size, pred_seq_len, 1)
    '''
    # encoder test
    model = Encoder(input_size=feature,hidden_size=5)
    hidden, cell = model(x)
    print(f"hidden.shape:{hidden.shape}")
    
    '''
    # seq2seq test
    model = RNN_Seq2Seq(input_size=feature,hidden_size=5,predict_seqlen=pred_seq_len)
    outputs = model(x)
    print(outputs.shape)
    
    
