import os
import torch
import sys
import time
import seq2seq
import numpy as np
import random
from utils.my_logging import My_Logging as My_Logging
import utils.prepare_data as prepare_data
from alive_progress import alive_bar
import matplotlib.pyplot as plt

# 定义超参数
HyperParams = {"batch_size":128,
                "N_EPOCHS":50,
                'lr':1e-3,
                "time_steps":14,
                "features":2,
                "horizion":7,
                "predict_seqlen":7
                }

DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
torch.set_default_tensor_type('torch.DoubleTensor')

# 配置Logging
my_log = My_Logging()

class Train():
    def __init__(self, hyperparams, model, optimizer, loss_func):

        # 超参数
        self.N_EPOCHS = hyperparams['N_EPOCHS']
        self.lr = hyperparams['lr']
        self.batch_size = hyperparams['batch_size']
        self.time_steps = hyperparams['time_steps']
        self.horizion = hyperparams['horizion']
        self.predict_seqlen = hyperparams['predict_seqlen']
        self.features = hyperparams['features']

        # 数据生成器
        self.train_generator = None
        self.test_generator = None

        # 模型、优化器、损失函数
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func

        # 训练过程loss值
        self.train_loss = []
        self.valid_loss = []

        # 自动设置 seed
        self._setup_seed(20)
        # 自动生成数据生成器
        self._prepare_data()
    
    def _prepare_data(self):
        train_ip, train_op, test_ip, test_op = prepare_data.prepare_data(time_steps=self.time_steps, horizion=self.horizion, features=self.features)
        train_dataset = prepare_data.My_Train_Datasets(train_ip, train_op)
        test_dataset = prepare_data.My_Test_Dataset(test_ip, test_op)

        train_generator = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False)
        test_generator = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        self.train_generator = train_generator
        self.test_generator = test_generator
        
    def save_state(self, state,filename="my_state.pt"):
        my_log.info_logger("Saving model and optimizer state")
        torch.save(state, filename)

    def load_state(self, filename="my_state.pt"):
        my_log.info_logger("Loading model and optimizer state")
        print("Loading model and optimizer state")
        self.model.load_state_dict(torch.load(filename)['model'])
        self.optimizer.load_state_dict(torch.load(filename)['optimizer'])
        self.train_loss = torch.load(filename)['train_loss']
        self.valid_loss = torch.load(filename)['valid_loss']

    def _train(self):
        '''
        params:
                None
        returns:
            每个样本的平均损失值
        '''
        self.model.train()
        epoch_loss = 0
        
        for i, (x,y) in enumerate(self.train_generator):
            x = x.to(DEVICE)  # [batch_size, seq_len, feature]
            y = y.to(DEVICE)  # [batch_size, predict_seqlen]

            self.optimizer.zero_grad()
            # 输入到模型的是[batch_size, seq_len, features],模型输出是[batch_size,predict_seqlen, 1]
            output = self.model(x)

            loss = self.loss_func(output.squeeze(dim=2), y) # 将output [batch_size, predict_seqlen]
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        # len(train_generator) 是 批次数量  即 样本总数/batch_size
        return epoch_loss / len(self.train_generator)

    def _evalute(self):
        '''
        模型评估函数
        params:
            无
        return: 损失均值
        '''
        self.model.eval()
        epoch_loss = 0
        for i,(x,y) in enumerate(self.test_generator):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            with torch.no_grad():
                output = self.model(x) # output [batch_size, predict_seqlen, 1]
                loss = self.loss_func(output.squeeze(dim=2), y)
                epoch_loss += loss.item()
        return epoch_loss / len(self.test_generator)

    def _epoch_time(self, start_time, end_time):
        '''
        function: calculate the time of every epoch
        params:
            start_time: 
            end_time
        return:
            elapsed_mins:
            elapsed_secs:
        '''
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time/60)
        elapsed_secs = int(elapsed_time - elapsed_mins * 60)
        return elapsed_mins, elapsed_secs    
    
    def _setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def train_model(self):  
        # 初始最佳loss值为无穷大
        best_valid_loss = float('inf')
        # 训练进度条
        with alive_bar(total=self.N_EPOCHS, title='training') as bar:
            for epoch in range(self.N_EPOCHS):
                # 记录开始时间
                start_time = time.time()
                
                # 训练评估
                train_loss = self._train()
                valid_loss = self._evalute()
                
                # 将 训练评估值保存起来
                self.train_loss.append(train_loss)
                self.valid_loss.append(valid_loss)
                
                # 记录结束时间
                end_time = time.time()
                # 计算 分钟、秒
                epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)

                # 保存最好的模型
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    my_state = {'model':self.model.state_dict(), 
                                "optimizer":self.optimizer.state_dict(), 
                                "train_loss":self.train_loss, 
                                "valid_loss":self.valid_loss}
                    self.save_state(state=my_state)
                
                # 打印该epoch训练信息
                print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.4f}')
                print(f'\t Val. Loss: {valid_loss:.4f}')

                # 更新进度条
                bar()

    def plot_loss(self,save:bool=False):
        # x = [i for i in np.linspace(0,self.N_EPOCHS)]
        fig = plt.figure(num=1, figsize=(15, 8),dpi=80)     #开启一个窗口，同时设置大小，分辨率
        ax1 = fig.add_subplot(1,1,1)                        #通过fig添加子图，参数：行数，列数，第几个
        ax1.set_title('loss_value')                        #设置图体，plt.title
        ax1.set_xlabel('epoch')                     #设置x轴名称,plt.xlabel
        ax1.set_ylabel('loss')                    #设置y轴名称,plt.ylabel
        ax1.set_xlim(0,self.N_EPOCHS)              #设置横轴范围，会覆盖上面的横坐标,plt.xlim
        # ax1.set_ylim(0,3)                         #设置纵轴范围，会覆盖上面的纵坐标,plt.ylim
        plot1=ax1.plot(self.train_loss, marker='o', color='g', label='train_loss')   #点图：marker图标
        plot2=ax1.plot(self.valid_loss, linestyle='--', alpha=0.5, color='r', label='valid_loss')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本
        ax1.legend(loc='upper left')            #显示图例,plt.legend()
        if save:
            plt.savefig('loss_value.jpg',dpi=400,bbox_inches='tight')   #savefig保存图片，dpi分辨率，bbox_inches子图周边白色空间的大小
        plt.show()    

    def show_example(self):
        '''
        展示训练后的模型在一个样本上的预测能力
        以 test_generation 的第一个样本为例
        '''
        for x,y in self.test_generator:
            x = x.to(DEVICE)  # [batch_size, seqlen,  feature]
            y = y.to(DEVICE)  # [batch_size, predict_seqlen, 1]
            break
        # 取第一个批次的第一个样本
        inputs = x[0].unsqueeze(dim=0)  # [batch_size=1, seqlen, feature]
        target = y[0]                    # [seq_len=predict_seqlen, feature=1]

        pred = self.predict(inputs)     # [batch_szie=1, predict_seqlen, features]
        pred = pred.squeeze(dim=0)       # [predict_seqlen, features=1]
        fig = plt.figure()

        
        # 将输入和 target 合并一起
        inputs_targets = torch.cat((inputs[:,:,0].squeeze(dim=0), target.reshape(-1)))
        # 将输入和 pred 合并一起
        inputs_pred = torch.cat((inputs[:,:,0].squeeze(dim=0), pred.reshape(-1)))
        
        # 在预测的点上做标记
        markers_on = np.arange(pred.shape[0]) + inputs.shape[1]  # 14 + [0,1,2,3]

        plt.plot(inputs_targets.cpu().numpy().reshape(-1),label="target")
        plt.plot(inputs_pred.cpu().numpy().reshape(-1),label="pred", marker='D', markevery=markers_on)
        plt.legend()
        plt.show()
        
    def predict(self, inputs):
        self.model.eval()
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            output = self.model(inputs) # output [batch_size, predict_seqlen, 1]
        return output

def main():
    # 是否加载模型
    load_model = True

    # 模型 优化器 损失函数
    model = seq2seq.RNN_Seq2Seq(input_size=HyperParams['features'], hidden_size=5, predict_seqlen=HyperParams['predict_seqlen']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=HyperParams['lr'])
    loss_func = torch.nn.MSELoss(reduction='mean')

    T = Train(hyperparams=HyperParams, model=model, optimizer=optimizer, loss_func=loss_func)
    if load_model:
        T.load_state()  # 加载之前训练好的模型
    else:
        T.train_model() # 再训练一次
    T.plot_loss()
    T.show_example()

if __name__ == "__main__":
    main()