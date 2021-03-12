import logging

class My_Logging:
    '''
    配置日志信息
    '''
    def __init__(self, logname='log.log'):

        logging.basicConfig(filename=logname, 
                        filemode='a',
                        level=logging.INFO,         
                        format="%(asctime)s--%(filename)s|-- %(message)s",   
					    datefmt= "%Y-%m-%d %H:%M:%S")  

    def debug_logger(self, log_text):
        logging.debug(log_text)
    def info_logger(self, log_text):
        logging.info(log_text)
    def error_logger(self, log_text):
        logging.error(log_text)