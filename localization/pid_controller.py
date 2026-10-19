import collections
class PIDController():
    def __init__(self, kp : float, ki : float, kd : float, error_history_length : int = 100):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.cumulated_error = 0.0
        self.last_error = 0.0
        self.error_history = collections.deque(maxlen=error_history_length)

    def update(self, e):
        self.error_history.append(e)
        self.cumulated_error = sum(self.error_history)

        u = self.kp * e + self.ki * self.cumulated_error - self.kd * (e - self.last_error)

        self.last_error = e
        return u