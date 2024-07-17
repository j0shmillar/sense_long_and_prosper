import gym
import csv
import pandas 
import argparse
import numpy as np
from rich.progress import track

ACTIONS = [3, 5, 10, 60, 300, 600, 1800]

def s2h(s):
    return int(((s // 3600) + args.start_time) % 24)

def write_results(filename, data_to_append):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_append)

def handle_event(t, event_starts, event_ends):
    ongoing_events = (event_starts <= t + args.t_int) & (event_ends >= t)
    detected_events = np.sum(ongoing_events)
    if detected_events > 0:
        t_end = np.max(event_ends[ongoing_events])
        return t_end, detected_events
    return t + args.t_int, 0

class EventCaptureEnv(gym.Env):
    def __init__(self, event_data, max_episodes=1000, max_steps_per_episode=24):
        super(EventCaptureEnv, self).__init__()

        self.event_data = event_data
        self.num_hours = max_steps_per_episode
    
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Discrete(self.num_hours)

        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.3
        self.Q = np.zeros((self.num_hours, len(ACTIONS))) 

        self.current_hour = 0
        self.total_events_captured = 0
        self.total_active_time = 0
        self.T_INT = args.t_int

        self.current_episode = 0
        self.current_step = 0

    def reset(self):
        self.current_hour = 0
        self.total_events_captured = 0
        self.total_active_time = 0
        self.current_episode += 1
        self.current_step = 0
        return self.current_hour

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(ACTIONS))
        else:
            return np.argmax(self.Q[self.current_hour])

    def step(self, action, t, w):
        self.epsilon = max(0.1, self.epsilon * 0.99) 
        hourly_wake_up_rate = ACTIONS[action]
        event_starts = np.array(self.event_data['start_time'])
        event_ends = np.array(self.event_data['end_time'])
        active_time = 0
        total_events_captured = 0
        pos_wake_up, neg_wake_up = 0,0
        while s2h(t)==self.current_hour:
            t_old = t
            t, events_captured = handle_event(t, event_starts, event_ends)
            if events_captured>0:
                pos_wake_up+=1
            else:
                neg_wake_up+=1
            if t>t_old:
                active_time+=t-t_old 
            else:
                active_time+=self.T_INT
                t+=self.T_INT
            total_events_captured+=events_captured
            t+=hourly_wake_up_rate

        reward = total_events_captured - w*neg_wake_up

        self.total_events_captured+=total_events_captured
        self.total_active_time+=active_time
        prev_hour = self.current_hour
        self.current_hour = s2h(t)
        self.Q[prev_hour, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.Q[self.current_hour%24, next_action]) - self.Q[prev_hour, action])
        done = self.current_hour > self.num_hours or self.current_hour==0 or self.current_step > self.max_steps_per_episode
        self.current_step += 1
        return self.current_hour, reward, done, {}, t
    
    def save_q_table(self, file_path):
        np.save(file_path, self.Q)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def fixed(data, START, END, int_action, t_int):
    t = START
    event_starts = np.array(data['start_time'])
    event_ends = np.array(data['end_time'])
    active_time, total_events_detected = 0, 0
    pos_wake_up, neg_wake_up = 0, 0
    while t <= END - t_int:
        t_old = t
        t, events_detected = handle_event(t, event_starts, event_ends)
        total_events_detected += events_detected
        if events_detected>0:
            pos_wake_up+=1
        else:
            neg_wake_up+=1
        if t - t_old > t_int:
            active_time += t - t_old
        else:
            t += t_int
            active_time += t_int
        t += int_action
    return active_time, total_events_detected, pos_wake_up/(pos_wake_up+neg_wake_up), (pos_wake_up+neg_wake_up)

def eval(data, q_table, START, END, t_int):
    t = START
    current_hour = s2h(START)
    event_starts = np.array(data['start_time'])
    event_ends = np.array(data['end_time'])
    active_time, total_events_detected = 0, 0
    pos_wake_up, neg_wake_up = 0, 0
    while t <= END - t_int:
        action = ACTIONS[np.argmax(q_table[current_hour])]
        while s2h(t)==current_hour:
            t_old = t
            t, events_detected = handle_event(t, event_starts, event_ends)
            if events_detected>0:
                pos_wake_up+=1
            else:
                neg_wake_up+=1
            if t - t_old > t_int:
                active_time += t - t_old
            else:
                t += t_int
                active_time += t_int
            total_events_detected+=events_detected
            t+=action
        current_hour = s2h(t)
    return active_time, total_events_detected, pos_wake_up/(pos_wake_up+neg_wake_up), (pos_wake_up+neg_wake_up) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # add epsilion, decay, others
    parser.add_argument("--load", default=0, type=int, help='use existing q-table')
    parser.add_argument("--events", default='../../../../event_times1.csv', type=str, help='path to event data')
    parser.add_argument("--path", default='q_table.npy', type=str, help='q-table path')
    parser.add_argument("--t_hrs", default=24, type=int, help='train hours')
    parser.add_argument("--v_hrs", default=24, type=int, help='eval hours')
    parser.add_argument("--t_int", default=0.1, type=float, help='wake duration')
    parser.add_argument("--weight", default=0.0001, type=float, help='balancing weight')
    parser.add_argument("--n_runs", default=10, type=int, help='number of runs')
    parser.add_argument("--n_episodes", default=1000, type=int, help='number of episodes')
    parser.add_argument("--start_time", default=0.5, type=float, help='start time') # change
    args = parser.parse_args()

    for _ in range(args.n_runs):

        data = pandas.read_csv(args.events)
        data = data[data['confidence']>=0.7]
        TRAIN = args.t_hrs * 60 * 60
        data_train = data[(data['start_time'] < TRAIN)]
        EVAL = TRAIN + args.v_hrs * 60 * 60
        data_eval = data[(data['start_time'] >= TRAIN) & (data['start_time'] < EVAL)]
        EVAL_START = TRAIN
        # EVAL_END = EVAL
        EVAL_END = max(data_eval['end_time'])

        env = EventCaptureEnv(data_train)
        num_episodes = 100

        if args.load == 0:
            for episode in track(range(num_episodes), "Learning..."):
                observation = env.reset()
                done = False
                t = 0
                while not done:
                    action = env.choose_action()
                    observation, reward, done, _, t = env.step(action, t, args.weight)
                q_table_file = args.path
            env.save_q_table(q_table_file)
            q_table = np.load(args.path)
        else:
            q_table = np.load(args.path)

        total_events = len(data_eval)
        active_time_cont = EVAL_END - EVAL_START

        active_time_eval, total_events_eval, wake_up_ratio, n_wake_ups = eval(data_eval, q_table, EVAL_START, EVAL_END, args.t_int)

        print(f"Continuous:\nActive Time = {active_time_cont}, Detected Events = {total_events}")

        active_time_3s, total_events_detected_fixed_3s, wake_up_ratio_3s, n_wake_ups_3s = 0,0,0,0
        for i in [3, 5, 7]:
            active_time_fixed, total_events_detected_fixed, wake_up_ratio, n_wake_ups_fixed = fixed(data_eval, EVAL_START, EVAL_END, i, args.t_int)
            if i == 3:
                active_time_3s = active_time_fixed
                total_events_detected_fixed_3s = total_events_detected_fixed
                n_wake_ups_3s = n_wake_ups_fixed
                print(f"Fixed {i}s:\nActive Time = {active_time_fixed:.0f}, Detected Events = {total_events_detected_fixed}, Detected Events % = {total_events_detected_fixed/len(data_eval)*100:.2f}")
                print(f"N Wake ups = {n_wake_ups_fixed}")
                data_to_append = [active_time_fixed, total_events_detected_fixed/len(data_eval), n_wake_ups_fixed]
                write_results(f'fixed_{i}.csv', data_to_append)     
            else:
                print(f"Fixed {i}s:\nActive Time = {active_time_fixed:.0f}, Active Time % = {(active_time_fixed/active_time_3s)*100:.2f}, Detected Events = {total_events_detected_fixed}, Detected Events % = {(total_events_detected_fixed/len(data_eval))*100:.2f}")
                print(f"N Wake ups = {n_wake_ups_fixed}")
                print(f"N Wakes ups % = {((n_wake_ups_fixed)/n_wake_ups_3s)*100:.2f}")
                data_to_append = [active_time_fixed, (active_time_fixed/active_time_3s)*100, (total_events_detected_fixed/len(data_eval))*100, (n_wake_ups_fixed/n_wake_ups_3s)*100]
                write_results(f'fixed_{i}.csv', data_to_append)

        print(f"QL:\nActive Time = {active_time_eval:.0f}, Active Time % = {(active_time_eval/active_time_3s)*100:.2f}, Detected Events = {total_events_eval}, Detected Events % = {(total_events_eval/len(data_eval))*100:.2f}")
        print(f"N Wake ups = {n_wake_ups}")
        print(f"N Wakes ups % = {((n_wake_ups)/n_wake_ups_3s)*100:.2f}")

        data_to_append = [active_time_eval, (active_time_eval/active_time_3s)*100, (total_events_eval/len(data_eval))*100, (n_wake_ups/n_wake_ups_3s)*100]
        write_results(f'ql.csv', data_to_append)

        env.close()


