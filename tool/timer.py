import time

time_dict = {}

def timer_start(name):
    time_dict[name] = time.time()

def timer_end(name) -> float:
    time_dict[name] = time.time() - time_dict[name]
    return time_dict[name]

def timer_get(name) -> float:
    return time_dict[name]

def timer_print(name):
    t = timer_end(name)
    print(f"\n[DEBUG] [Timer] {name}\t{t:4f}")
