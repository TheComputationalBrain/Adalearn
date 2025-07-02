# Imports
import tkinter as tk
import time

# Task and stimulation parameters
# VNS parameters (in seconds)
Stim_ON = 30
Stim_OFF = 300
Stim_total_dur = Stim_ON + Stim_OFF

# Ada_Learn parameters
Num_stimuli = 75
ISI = 1.5
Task_session_dur = Num_stimuli * ISI

Int_pre_stim = (Task_session_dur - Stim_ON) / 2

Task_session_interval = (Stim_total_dur - (Stim_ON + Int_pre_stim + Task_session_dur + Int_pre_stim)) / 2
Task_session_cycle = Task_session_dur + Task_session_interval

# Task structure relative to VNS stimulation
VNS_A = Stim_total_dur - Int_pre_stim
Lower_VNS_A = VNS_A - 3
Higher_VNS_A = VNS_A + 3

VNS_B = (Stim_total_dur - (Task_session_dur + Task_session_interval + Int_pre_stim))
Lower_VNS_B = VNS_B - 3
Higher_VNS_B = VNS_B + 3


# VNS Timer
class Timer(tk.Toplevel):  # Use Toplevel instead of Tk
    def __init__(self, parent):
        super().__init__(parent)
        self.title("VNS TIMER")
        self.geometry("300x300")
        self.time_var = tk.StringVar()
        self.time_var.set("00:00:00")
        self.label = None
        self.start_time = time.time()
        self.create_widgets()
        self.update_time()

    def create_widgets(self):
        self.label = tk.Label(self, textvariable=self.time_var, font=("Arial", 60), bg="black")
        self.configure(background="black")
        self.label.pack(expand=True)

        self.reset_button = tk.Button(self, text="Reset", command=self.reset)
        self.reset_button.pack(expand=True)

    def reset(self):
        self.start_time = time.time()
        self.update_time()

    def update_time(self):
        elapsed_time = time.time() - self.start_time
        self.time_var.set(self.format_time(elapsed_time))
        if self.should_change_color(elapsed_time, Lower_VNS_A, Higher_VNS_A):
            self.change_color("orange")
        elif self.should_change_color(elapsed_time, Lower_VNS_B, Higher_VNS_B):
            self.change_color("purple")
        else:
            self.change_color("white")
        self.after(1000, self.update_time)

    def format_time(self, elapsed_time):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def should_change_color(self, elapsed_time, start_range, end_range):
        time_within_range = start_range <= elapsed_time % Stim_total_dur <= end_range
        return time_within_range

    def change_color(self, color):
        self.label.config(fg=color)

class Exp_Structure(tk.Toplevel):  # Use Toplevel instead of Tk
    def __init__(self, parent):
        super().__init__(parent)
        self.title("EXPERIMENT STRUCTURE")
        self.geometry("300x300")
        self.time_var = tk.StringVar()
        self.time_var.set("00:00:00")
        self.label = None
        self.start_time = time.time()
        self.create_widgets()
        self.update_time()

    def create_widgets(self):
        self.label = tk.Label(self, textvariable=self.time_var, font=("Arial", 60), bg="black")
        self.configure(background="black")
        self.label.pack(expand=True)

        self.reset_button = tk.Button(self, text="Reset", command=self.reset)
        self.reset_button.pack(expand=True)

    def reset(self):
        self.start_time = time.time()
        self.update_time()

    def update_time(self):
        elapsed_time = time.time() - self.start_time
        self.time_var.set(self.format_time(elapsed_time))
        if self.should_change_color(elapsed_time, Task_session_dur, Task_session_cycle):
            self.change_color("red")
        else:
            self.change_color("green")
        self.after(1000, self.update_time)

    def format_time(self, elapsed_time):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def should_change_color(self, elapsed_time, start_range, end_range):
        time_within_range = start_range <= elapsed_time % Task_session_cycle <= end_range
        return time_within_range

    def change_color(self, color):
        self.label.config(fg=color)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("App")
        self.timer = Timer(self)
        self.structure = Exp_Structure(self)

if __name__ == "__main__":
    app = App()
    app.mainloop()