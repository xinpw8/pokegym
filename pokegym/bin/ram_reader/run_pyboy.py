from os.path import exists
from pyboy import PyBoy, WindowEvent
from threading import Thread, Barrier
import random
import traceback

class PyBoyMgr:
    def __init__(self, rom_path, barrier):
        self._pyboy = PyBoy(rom_path, debugging=False, disable_input=False, window_type='headless')
        self.barrier = barrier

    def run_pyboy(self, thread_id):
        try:
            frame = 0
            for i in range(random.randint(12, 30)):
                self._pyboy.tick()
                frame += 1

            print(f"Thread {thread_id} finished after {frame} frames")
        except Exception as e:
            print(f"Error in run_pyboy for thread {thread_id}: {e}")
            traceback.print_exc()

    def thread_function(self, thread_id):
        try:
            while True:
                self.run_pyboy(thread_id)
                self.barrier.wait()  # Wait for all threads at the barrier
        except Exception as e:
            print(f"Error in thread_function for thread {thread_id}: {e}")
            traceback.print_exc()
        finally:
            self._pyboy.stop()

def start_threads(num_threads, rom_path):
    barrier = Barrier(num_threads)
    threads = []
    managers = []

    for i in range(num_threads):
        mgr = PyBoyMgr(rom_path, barrier)
        managers.append(mgr)
        thread = Thread(target=mgr.thread_function, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("All threads have finished.")

# Example usage
start_threads(120, '../PokemonRed.gb')
