import threading
import queue
import time
from network import  push_data_worker,push_data_async
import random


if __name__ == '__main__':
   
   work_thread = threading.Thread(target= push_data_worker)
   work_thread.isDaemon = True
   work_thread.start()
#    work_thread.join()
   i = 0
   while True:
        score_data = {
                        "x": random.randint(0,200),
                        "y": random.randint(0,200),
                        "score": random.randint(10,50),
                        "device_id": random.randint(1000,2000),
                        "target_id": random.randint(5000,60000),
                    }
                    
        push_data_async(score_data)
        time.sleep(2)
      