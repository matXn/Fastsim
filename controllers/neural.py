"""Map-aware neural tracking reference trained with BC and DAgger."""

from pathlib import Path

import numpy as np

from controllers.planner import plan
from fastsim import PathFollower

DISTANCES = np.array([0.2, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0])
DIM = 21
SCHEMA = 'known_map_tracking_v2_lat03'
# Fixed feature/label scales are part of the checkpoint schema.
SPEED_SCALE = 6.0
YAW_RATE_SCALE = 8.0
ACCEL_SCALE = 4.0


class Features:
    def __init__(self, track, params, cap=6.0):
        path, speed = plan(track, params)
        self.teacher = PathFollower(path, np.minimum(speed, cap), params)
        self.params = params

    def query(self, state):
        speed, steer = self.teacher(state)
        follower = self.teacher
        indices = (follower.idx + np.maximum(1, (DISTANCES / follower.ds).astype(int))) % len(follower.path)
        delta = follower.path[indices] - (state.x, state.y)
        cosine, sine = np.cos(state.yaw), np.sin(state.yaw)
        local = np.column_stack((delta[:, 0] * cosine + delta[:, 1] * sine,
                                 -delta[:, 0] * sine + delta[:, 1] * cosine)) / 5.0
        observation = np.concatenate((local.ravel(), [
            state.v / SPEED_SCALE,
            state.steer / self.params.max_steer,
            state.yaw_rate / YAW_RATE_SCALE,
            state.accel / ACCEL_SCALE,
            speed / SPEED_SCALE,
        ])).astype(np.float32)
        label = np.array([speed / SPEED_SCALE, steer / self.params.max_steer], np.float32)
        return observation, label


class Net:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.w = [(rng.normal(size=(inputs, outputs)) * np.sqrt(2 / inputs)).astype(np.float32)
                  for inputs, outputs in [(DIM, 64), (64, 64), (64, 2)]]
        self.b = [np.zeros(size, np.float32) for size in [64, 64, 2]]
        self.m = [np.zeros_like(parameter) for parameter in self.w + self.b]
        self.v = [np.zeros_like(parameter) for parameter in self.w + self.b]
        self.step = 0

    def predict(self, observation):
        hidden = np.maximum(observation @ self.w[0] + self.b[0], 0)
        hidden = np.maximum(hidden @ self.w[1] + self.b[1], 0)
        return hidden @ self.w[2] + self.b[2]

    def fit(self, observations, labels, updates=500, seed=0):
        rng = np.random.default_rng(seed)
        loss = 0.0
        for _ in range(updates):
            indices = rng.integers(len(observations), size=min(512, len(observations)))
            batch, targets = observations[indices], labels[indices]
            hidden1 = np.maximum(batch @ self.w[0] + self.b[0], 0)
            hidden2 = np.maximum(hidden1 @ self.w[1] + self.b[1], 0)
            predictions = hidden2 @ self.w[2] + self.b[2]
            error = (predictions - targets) * np.array([1, 2], np.float32)
            loss = float(np.mean(error ** 2))
            gradient = 2 * error * np.array([1, 2], np.float32) / len(batch)
            weight2, bias2 = hidden2.T @ gradient, gradient.sum(axis=0)
            gradient = (gradient @ self.w[2].T) * (hidden2 > 0)
            weight1, bias1 = hidden1.T @ gradient, gradient.sum(axis=0)
            gradient = (gradient @ self.w[1].T) * (hidden1 > 0)
            weight0, bias0 = batch.T @ gradient, gradient.sum(axis=0)
            self.step += 1
            gradients = [weight0, weight1, weight2, bias0, bias1, bias2]
            for index, (parameter, gradient) in enumerate(zip(self.w + self.b, gradients)):
                gradient = np.clip(gradient, -5, 5)
                self.m[index] = 0.9 * self.m[index] + 0.1 * gradient
                self.v[index] = 0.999 * self.v[index] + 0.001 * gradient * gradient
                moment = self.m[index] / (1 - 0.9 ** self.step)
                variance = self.v[index] / (1 - 0.999 ** self.step)
                parameter -= 0.001 * moment / (np.sqrt(variance) + 1e-8)
        return loss

    def save(self, path, cap):
        np.savez(path, **{f'w{index}': weight for index, weight in enumerate(self.w)},
                 **{f'b{index}': bias for index, bias in enumerate(self.b)}, cap=cap, schema=SCHEMA)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data['schema']) != SCHEMA:
                raise ValueError('Unsupported neural tracking schema')
            net = cls()
            net.w = [data[f'w{index}'].copy() for index in range(3)]
            net.b = [data[f'b{index}'].copy() for index in range(3)]
            return net, float(data['cap'])


class Controller:
    def __init__(self, track, car_params):
        self.net, self.cap = Net.load(Path(__file__).with_name('model.npz'))
        self.features = Features(track, car_params, self.cap)
        self.params = car_params

    def __call__(self, state):
        observation, _ = self.features.query(state)
        output = self.net.predict(observation)
        return (float(np.clip(output[0] * SPEED_SCALE, 0, self.cap)),
                float(np.clip(output[1], -1, 1) * self.params.max_steer))
