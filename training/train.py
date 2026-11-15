"""Train or evaluate the known-map BC + DAgger reference controller."""

import os

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from controllers.neural import DIM, SCHEMA, SPEED_SCALE, Features, Net
from fastsim import TRACKS, CarParams, load_track, run_lap

MODEL = Path(__file__).resolve().parents[1] / 'controllers' / 'model.npz'
HOLDOUT_SEEDS = [None, 90001, 90002, 90003]


def episode(name, seed, cap, net=None, beta=0.0, collect=False):
    params = CarParams()
    base = load_track(name)
    track = base if seed is None else base.perturbed(seed)
    features = Features(track, params, cap)
    rng = np.random.default_rng((seed or 0) + 789)
    observations, labels = [], []

    def control(state):
        observation, label = features.query(state)
        if collect:
            observations.append(observation)
            labels.append(label)
        output = label if net is None or rng.random() < beta else net.predict(observation)
        return (float(np.clip(output[0] * SPEED_SCALE, 0, cap)),
                float(np.clip(output[1], -1, 1) * params.max_steer))

    result = run_lap(track, control, params, noise_seed=seed)
    row = dict(track=track.name, seed=seed, cap=cap, finished=result.finished,
               reason=result.reason, time=result.time, progress=result.progress,
               ctrl_ms=result.ctrl_time * 1000)
    return (row, np.asarray(observations, np.float32).reshape(-1, DIM),
            np.asarray(labels, np.float32).reshape(-1, 2))


def evaluate(net, cap, seeds):
    return [episode(name, seed, cap, net)[0] for name in TRACKS for seed in seeds]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def smoke(out):
    row, observations, labels = episode('B', None, 1.5, collect=True)
    assert row['finished'] and np.isfinite(observations).all() and np.isfinite(labels).all()
    net = Net()
    before = float(np.mean((net.predict(observations) - labels) ** 2))
    net.fit(observations, labels, 100)
    after = float(np.mean((net.predict(observations) - labels) ** 2))
    assert after < before
    net.save(out / 'smoke.npz', 1.5)
    restored, _ = Net.load(out / 'smoke.npz')
    np.testing.assert_allclose(restored.predict(observations[:10]), net.predict(observations[:10]))

    batch, targets = observations[:4], labels[:4]
    predictions = net.predict(batch)

    def loss():
        return float(np.sum((net.predict(batch) - targets) ** 2 * np.array([1, 4])) / len(batch))

    epsilon = 1e-3
    original = float(net.b[2][1])
    net.b[2][1] = original + epsilon
    plus = loss()
    net.b[2][1] = original - epsilon
    minus = loss()
    net.b[2][1] = original
    analytic = float((2 * (predictions - targets) * np.array([1, 4]) / len(batch)).sum(axis=0)[1])
    assert abs((plus - minus) / (2 * epsilon) - analytic) < 1e-3
    print(json.dumps(dict(smoke='passed', samples=len(observations), loss_before=before, loss_after=after)))


def train(out, rounds, updates, caps):
    teacher = evaluate(None, caps[-1], [None, 50001, 50002])
    write_json(out / 'teacher_baseline.json', teacher)
    print('TEACHER', sum(row['finished'] for row in teacher), '/', len(teacher), flush=True)
    if not all(row['finished'] for row in teacher):
        raise RuntimeError('Teacher baseline failed')

    net = Net(42)
    observations, labels = [], []
    stage, promoted = 0, 0
    best = {}
    for round_index in range(rounds):
        cap = caps[stage]
        beta = 1.0 if round_index == 0 else max(0.1, 0.6 * 0.8 ** round_index)
        for index, name in enumerate(TRACKS):
            for seed in (None, 10000 + round_index * 100 + index):
                _, batch, targets = episode(name, seed, cap, net, beta, collect=True)
                observations.append(batch)
                labels.append(targets)
        batch = np.concatenate(observations)[-160000:]
        targets = np.concatenate(labels)[-160000:]
        observations, labels = [batch], [targets]
        loss = net.fit(batch, targets, updates, round_index)
        rows = evaluate(net, cap, [None, 50001, 50002])
        success = sum(row['finished'] for row in rows)
        score = (success, -sum(row['time'] for row in rows if row['finished']))
        net.save(out / f'candidate_{round_index:03d}.npz', cap)
        if stage not in best or score > best[stage]:
            best[stage] = score
            net.save(out / f'best_cap_{cap:g}.npz', cap)
        report = dict(round=round_index, stage=stage, cap=cap, beta=beta, samples=len(batch),
                      loss=loss, finished=success, total=len(rows), eval=rows)
        with (out / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(report) + '\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'eval'}), flush=True)
        promoted = promoted + 1 if success == len(rows) else 0
        if promoted >= 2 and stage < len(caps) - 1:
            stage += 1
            promoted = 0

    chosen = out / f'best_cap_{caps[stage]:g}.npz'
    if not chosen.exists():
        chosen = out / f'best_cap_{caps[max(0, stage - 1)]:g}.npz'
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='检查采样、训练梯度和模型读写')
    mode.add_argument('--evaluate', action='store_true', help='在独立测试种子上复测模型')
    parser.add_argument('--model', type=Path, default=MODEL, help='复测模型路径')
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--out', type=Path, help='本次运行输出目录，默认在 runs/ 下新建')
    args = parser.parse_args()
    if args.rounds < 1 or args.updates < 1:
        parser.error('--rounds and --updates must be positive')
    out = args.out or Path('runs') / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    caps = [min(cap, CarParams().v_max) for cap in [1.5, 2.5, 4.0, 6.0]]
    config = dict(method='BC+DAgger', caps=caps, train_seeds='10000+round*100+track',
                  selection_seeds=[50001, 50002], holdout_seeds=HOLDOUT_SEEDS,
                  architecture=[DIM, 64, 64, 2], schema=SCHEMA, car_params=asdict(CarParams()),
                  rounds=args.rounds, updates=args.updates,
                  mode='train', note='Full map planner provides path and speed profile. Neural model learns tracking.')
    if args.smoke:
        config['mode'] = 'smoke'
        write_json(out / 'config.json', config)
        smoke(out)
        return
    if args.evaluate:
        config['mode'] = 'evaluate'
    write_json(out / 'config.json', config)
    chosen = args.model if args.evaluate else train(out, args.rounds, args.updates, caps)
    net, cap = Net.load(chosen)
    rows = evaluate(net, cap, HOLDOUT_SEEDS)
    write_json(out / 'holdout.json', rows)
    summary = dict(candidate=str(chosen), cap=cap, finished=sum(row['finished'] for row in rows), total=len(rows))
    write_json(out / 'summary.json', summary)
    print('FINAL', json.dumps(summary), flush=True)
    if not args.evaluate and all(row['finished'] for row in rows):
        # Each run owns its export so concurrent runs cannot replace the reference checkpoint.
        net.save(out / 'model.npz', cap)


if __name__ == '__main__':
    main()
