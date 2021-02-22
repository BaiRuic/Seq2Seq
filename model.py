import torch 
import torch.nn as nn

class Encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size:int, hidden_size:int, dropout=0.2):
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
                            batch_first=True, 
                            dropout=dropout)
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
        outputs, (hidden, cell) = self.lstm(x)
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
                            batch_first=True,
                            dropout=dropout)

        self.fc = nn.Linear(in_features=self.hidden_size, 
                            out_features=self.output_size, 
                            bias=True)

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
        outputs, (hidden, cell) = self.lstm(inputs,(prev_hidden, prev_cell))
        # prediction [batch, seq_len=1, outputs_size=1]
        prediction = self.fc(outputs.view(batch_size, self.hidden_size)).unsqueeze(dim=1)
       
        return prediction, hidden, cell 

class Seq2Seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    def __init__(self, input_size, hidden_size):
        super(Seq2Seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size=input_size,hidden_size=self.hidden_size)
        self.decoder = Decoder(input_size=1, hidden_size=self.hidden_size)

    def forward(self,  source, target):
        '''
        source [batch_size, seq_len, feature] 输入到编码器的数据
        target [batch_size, seq_len, 1] 和解码器输出计算loss的, 1是负载预测，只有一个输出
               target[:,0,:]为输出到解码器第一时间步的数据,所以应该和source最后一个时间步数据相关
        '''
        batch_size = source.shape[0]
        target_seq_len = target.shape[1]
        
        # outputs 存放最后解码器输出
        outputs = torch.zeros_like(target)

        # Encodering
        hidden, cell = self.encoder(source)
        
        # 解码器初始输出x 取Target第0个时间步
        x = target[:, 0, :].unsqueeze(dim=1)
        
        # Decodering
        for t in range(1, target_seq_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:,t,:] = output.squeeze(dim=1)
            x = output

        return outputs

if __name__ == '__main__':
    batch_size = 64
    seq_len = 14
    feature = 2
    pred_seq_len = 5
    x = torch.rand((batch_size, seq_len, feature))
    y = torch.rand(batch_size, pred_seq_len, 1)
    
    model = Seq2Seq(input_size=2,hidden_size=5)
    
   
    outputs = model(x,y)
    print(outputs.shape)
    '''
    
    pre_hidd, pre_cell = model_en(x)
    print('*' * 5 + 'encoder' + '*' *5)
    print(pre_hidd.shape, pre_cell.shape)

    print('*' * 5 + 'decoder' + '*' *5)
    model_de = Decoder(1,5)
    pred, hidden, cell = model_de(y, pre_hidd, pre_cell)

    print(pred.shape,hidden.shape, cell.shape)
    '''

    