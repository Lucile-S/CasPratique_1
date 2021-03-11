import os 
import logging
import time
from datetime import datetime


def current_date():
    # -- define the current time
    timeStamp = time.time()
    date = datetime.fromtimestamp(timeStamp).strftime("%Y-%m-%d")
    return date 

class Log:
    def __init__(self, log_file,date=current_date(), log_dir= os.path.join(os.getcwd(),'..', 'logs')):
        print(date)
        """ create log file """
        self.log_file = log_file
        self.date = date
        # --  define and create log directory if not exist in the current directory 
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # -- define logfile path  
        self.log_path =  os.path.join(self.log_dir, self.log_file + "_" + self.date +'.log')
    
    def get_logger(self,logger_name,level=logging.DEBUG):
        """ setup logging files """
        self.logger_name = logger_name
        self.level = level 
        # create logger named logger_name
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.level) 
        # define a format 
        formatter =  logging.Formatter('%(asctime)s-%(name)s- %(levelname)s  | %(message)s')
        # create file handler which logs even info messages 
        self.fileHandler = logging.FileHandler(self.log_path,'a', 'utf-8') 
        self.fileHandler.setLevel(logging.INFO)
        self.fileHandler.setFormatter(formatter)
        # create console handler
        self.streamHandler = logging.StreamHandler()
        self.streamHandler.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(self.fileHandler)
        self.logger.addHandler(self.streamHandler)
        return self.logger

    def read_logs(self):
    
        with open(self.log_path) as f:
            lines = f.read()
        return lines





# def create_log(log_file, path=os.getcwd()):
#         """ create log file """
#         # -- define the current time
#         timeStamp = time.time()
#         date = datetime.fromtimestamp(timeStamp).strftime("%Y-%m-%d")
#         # --  define and create log directory if not exist in the current directory 
#         log_dir = os.path.join(path, 'log')
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#         # -- define logfile 
#         log_path =  os.path.join(log_dir, log_file + "_" + date +'.log')
#         return log_path

# def setup_logger(log_path, level=logging.DEBUG):
#     format="%(asctime)s - %(name)s - %(levelname)s |  %(message)s"
#     return logging.basicConfig(filename=log_path, format=format,level=level)
