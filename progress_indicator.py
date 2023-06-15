import sentiment_packages as sp
import threading
import time

def print_dots(stop_progress_indicator):
	while not stop_progress_indicator.is_set():
		print('.', end='', flush=True)
		time.sleep(1)

# create an event object to signal the dot printing thread to stop
stop_progress_indicator = threading.Event()

# start the dot printing thread
dot_thread = threading.Thread(target=print_dots, args=(stop_progress_indicator,))

def start_thread():
	stop_progress_indicator.clear()
	dot_thread.start()

def stop_thread():
	stop_progress_indicator.set()
	dot_thread.join()