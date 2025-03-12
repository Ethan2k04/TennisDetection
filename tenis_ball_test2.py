import threading
import queue
import time

notify_server_event = threading.Event()
notify_server_queue = queue.Queue()

def worker():
    print("Worker thread waiting for event...")
    while True:
        # notify_server_event.wait()  # 阻塞，直到事件被设置
        print("Event received! Worker thread starts processing.")
        input  = notify_server_queue.get()
        print(input)


if __name__ == '__main__':
   
   work_thread = threading.Thread(target= worker)
   work_thread.isDaemon = True
   work_thread.start()
#    work_thread.join()
   i = 0
   while True:
       notify_server_queue.put({ 'x':1, 'y': 2})
    #    notify_server_event.set()
       time.sleep(2)
       i+=1