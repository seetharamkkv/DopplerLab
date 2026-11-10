#!/usr/bin/env python3
"""Run from DopplerLab. This never replaces the existing GRU scripts."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ebranch_tracker.core import discover, read_split, velocity_bins
from ebranch_tracker.data import Clips, collate, training_stats, read_labels
from ebranch_tracker.model import Tracker
from ebranch_tracker.losses import objective
from ebranch_tracker.evaluation import evaluate, move_batch


def choose_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', required=True, help='Existing explicit train/val/test sample lists')
    p.add_argument('--encoder', choices=['ebranchformer', 'gru'], default='ebranchformer')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--kernel', type=int, default=15)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--bin-width', type=float, default=1.)
    p.add_argument('--vmin', type=float, default=-60.)
    p.add_argument('--vmax', type=float, default=60.)
    p.add_argument('--corner-weight', type=float, default=3.)
    p.add_argument('--ordinal-weight', type=float, default=0.2)
    p.add_argument('--position-weight', type=float, default=0., help='Optional auxiliary loss; 0 for encoder comparison')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=50, help='Epochs since last improvement at a validation check')
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--warmup-epochs', type=int, default=10)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--sample-rate', type=int, default=44100)
    p.add_argument('--n-fft', type=int, default=2048)
    p.add_argument('--hop', type=int, default=512)
    p.add_argument('--speed-threshold', type=float, default=0.5)
    p.add_argument('--viterbi-penalty', type=float, default=0.05)
    p.add_argument('--max-jump', type=int, default=8)
    p.add_argument('--cache', help='Optional disk cache. Default: sibling shared feature cache')
    p.add_argument('--num-workers', type=int, default=0, help='0 is the portable macOS default')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true',
                   help='Continue this --out from last.pt (weights, Adam moments, best score, RNG).')
    return p.parse_args()


LOCKED = ('encoder', 'hidden', 'layers', 'heads', 'kernel', 'dropout', 'bin_width', 'vmin', 'vmax',
          'corner_weight', 'ordinal_weight', 'position_weight', 'seed', 'epochs', 'batch_size', 'lr',
          'warmup_epochs', 'weight_decay', 'viterbi_penalty', 'max_jump', 'sample_rate', 'n_fft',
          'hop', 'speed_threshold')


def save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_payload(device, generator):
    state = {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'torch_rng': torch.get_rng_state(), 'loader_rng': generator.get_state()}
    if device.type == 'cuda':
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    if device.type == 'mps' and hasattr(torch.mps, 'get_rng_state'):
        state['mps_rng'] = torch.mps.get_rng_state()
    return state


def restore_rng(checkpoint, device, generator):
    """Return True when the dataloader shuffle generator was restored exactly."""
    if 'python_rng' in checkpoint:
        random.setstate(checkpoint['python_rng'])
    if 'numpy_rng' in checkpoint:
        np.random.set_state(checkpoint['numpy_rng'])
    if 'torch_rng' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
    if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
    if device.type == 'mps' and checkpoint.get('mps_rng') is not None:
        torch.mps.set_rng_state(checkpoint['mps_rng'])
    if 'loader_rng' in checkpoint:
        generator.set_state(checkpoint['loader_rng'])
        return True
    return False


def advance_loader_rng(generator, n_items, epochs):
    """Replay PyTorch 2.14 DataLoader shuffle draws for num_workers=0.

    Each epoch draws one int64 seed, then two randperms (the second is discarded
    when the sample count divides the epoch length).
    """
    for _ in range(epochs):
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        torch.randperm(n_items, generator=generator)
        torch.randperm(n_items, generator=generator)


def recover_best(out, checkpoint):
    if 'best' in checkpoint and 'best_epoch' in checkpoint:
        return float(checkpoint['best']), int(checkpoint['best_epoch'])
    best, best_epoch = float('inf'), 0
    history = out / 'history.jsonl'
    if history.exists():
        for line in history.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ade = (row.get('val') or {}).get('ade_m')
            if ade is not None and ade < best:
                best, best_epoch = float(ade), int(row['epoch'])
    best_path = out / 'best.pt'
    if best_path.exists():
        saved = torch.load(best_path, map_location='cpu', weights_only=False)
        ade = (saved.get('val') or {}).get('ade_m')
        if ade is not None and ade <= best:
            best, best_epoch = float(ade), int(saved['epoch'])
    return best, best_epoch


def recover_elapsed(out, checkpoint):
    if 'elapsed_s' in checkpoint:
        return float(checkpoint['elapsed_s'])
    history = out / 'history.jsonl'
    elapsed = 0.
    if history.exists():
        for line in history.read_text().splitlines():
            if line.strip():
                elapsed = float(json.loads(line).get('elapsed_s', elapsed))
    return elapsed


def require_same_run(saved, args, root):
    if Path(saved['root']).resolve() != root:
        raise ValueError('Resume root does not match the saved run')
    for key in LOCKED:
        current, previous = getattr(args, key), saved[key]
        if isinstance(current, float) and isinstance(previous, float):
            if not math.isclose(current, previous, rel_tol=0, abs_tol=0):
                raise ValueError(f'Resume config mismatch: {key} ({current} != {previous})')
        elif current != previous:
            raise ValueError(f'Resume config mismatch: {key} ({current} != {previous})')


def main():
    args = arguments()
    if min(args.epochs, args.batch_size, args.eval_every, args.patience, args.threads) <= 0:
        raise ValueError('Epochs, batch size, evaluation interval, patience and threads must be positive')
    if args.lr <= 0 or args.corner_weight <= 0 or min(args.ordinal_weight, args.position_weight) < 0:
        raise ValueError('Invalid learning rate/loss weights')
    if args.viterbi_penalty < 0 or args.max_jump < 0:
        raise ValueError('Invalid decoder parameters')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    root, out = Path(args.root).expanduser().resolve(), Path(args.out).expanduser().resolve()
    last_path = out / 'last.pt'
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f'--resume needs {last_path}')
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty. Pass --resume to continue it, or choose a new --out.')
    available = discover(root)
    split = read_split(args.split, available)
    bins = velocity_bins(args.vmin, args.vmax, args.bin_width)
    feature_config = {'sample_rate': args.sample_rate, 'n_fft': args.n_fft, 'hop': args.hop,
                      'speed_threshold': args.speed_threshold}
    print(f'Device: {device}; PyTorch: {torch.__version__}', flush=True)
    print('Split: ' + ', '.join(f'{k}={len(v)}' for k, v in split.items()), flush=True)
    print('Checking ALL split labels and velocity coverage before training...', flush=True)
    selected = split['train'] + split['val'] + split['test']
    oracle_ades = []
    from ebranch_tracker.core import integrate
    for name in tqdm(selected, desc='Label/integration audit'):
        states, times, moving = read_labels(available[name], args.speed_threshold)
        velocities = states[:, [1, 3]]
        if velocities.min() < bins[0] - 1e-4 or velocities.max() > bins[-1] + 1e-4:
            raise ValueError(f'{name}: velocity outside [{bins[0]},{bins[-1]}]; widen bin bounds')
        positions = states[:, [0, 2]]
        oracle = integrate(velocities, times, positions[0])
        oracle_ades.append(float(np.linalg.norm(oracle[moving] - positions[moving], axis=1).mean()))
    print(f'Ground-truth velocity integration ADE: {np.mean(oracle_ades):.6f} m. '
          'Large values indicate label/clock inconsistency.', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(last_path, map_location='cpu', weights_only=False)
        saved_config = resume_checkpoint['config']
        require_same_run(saved_config, args, root)
        saved_split = json.loads((out / 'split.json').read_text())
        if saved_split != split:
            raise ValueError('Saved split.json does not match --split')
        config = saved_config
        print(f'Resuming after epoch {resume_checkpoint["epoch"]}', flush=True)
    else:
        config = vars(args).copy()
        config.update(root=str(root), out=str(out), features=feature_config,
                      model={'n_bins': len(bins), 'hidden': args.hidden, 'layers': args.layers,
                             'heads': args.heads, 'kernel': args.kernel, 'dropout': args.dropout,
                             'encoder': args.encoder},
                      source_split=str(Path(args.split).resolve()),
                      label_order=['x', 'vx', 'y', 'vy'],
                      oracle_integration_ade_m=float(np.mean(oracle_ades)),
                      implementation='standalone-ebranchformer-absolute-position-v1')
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        (out / 'split.json').write_text(json.dumps(split, indent=2))
    cache = Path(args.cache).expanduser().resolve() if args.cache else out.parent / 'ebranchformer_feature_cache'
    train = Clips(available, split['train'], feature_config, cache)
    stats_path = out / 'feature_stats.npz'
    if args.resume and stats_path.exists():
        saved_stats = np.load(stats_path)
        stats = {'mean': saved_stats['mean'], 'std': saved_stats['std'],
                 'frames': int(saved_stats['frames'])}
        print(f'Reusing feature statistics ({stats["frames"]} train frames)', flush=True)
    else:
        stats = training_stats(train)
        np.savez(stats_path, **stats)
    train.stats = stats
    val = Clips(available, split['val'], feature_config, cache, stats)
    test = Clips(available, split['test'], feature_config, cache, stats)
    generator = torch.Generator().manual_seed(args.seed)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    model = Tracker(**config['model']).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bins_tensor = torch.tensor(bins, device=device)
    best, best_epoch, start_epoch = float('inf'), 0, 1
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        if 'optimizer' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
        else:
            print('last.pt has no Adam moments; moments start over from these weights.', flush=True)
        best, best_epoch = recover_best(out, resume_checkpoint)
        start_epoch = int(resume_checkpoint['epoch']) + 1
        if not restore_rng(resume_checkpoint, device, generator):
            if args.num_workers != 0:
                raise RuntimeError('This last.pt has no shuffle state. Resume with --num-workers 0.')
            advance_loader_rng(generator, len(train), int(resume_checkpoint['epoch']))
            print(f'Shuffle generator replayed for {resume_checkpoint["epoch"]} finished epochs.', flush=True)
        print(f'Next epoch {start_epoch}; best val ADE {best:.4f} m at epoch {best_epoch}', flush=True)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **kwargs)
    val_loader, test_loader = DataLoader(val, **kwargs), DataLoader(test, **kwargs)
    start = time.time() - (recover_elapsed(out, resume_checkpoint) if resume_checkpoint else 0.)
    if start_epoch > args.epochs:
        raise RuntimeError(f'Training already finished {args.epochs} epochs. Nothing to resume.')
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            factor = epoch / max(1, args.warmup_epochs)
        else:
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            factor = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = args.lr * factor
        model.train()
        total_loss, samples = 0., 0
        bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in bar:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['features'], batch['lengths'])
            loss, parts = objective(logits, batch, bins_tensor, args.corner_weight,
                                    args.ordinal_weight, args.position_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss in {batch["ids"]}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            n = len(batch['ids'])
            total_loss += float(loss.detach()) * n
            samples += n
            bar.set_postfix(loss=f'{total_loss / samples:.4f}', lr=f'{args.lr * factor:.2g}')
        record = {'epoch': epoch, 'loss': total_loss / samples, 'elapsed_s': time.time() - start}
        is_check = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if is_check:
            summary, _ = evaluate(model, val_loader, bins, device, args.viterbi_penalty,
                                   args.max_jump, description='Validation Viterbi ADE')
            record['val'] = summary
            if summary['ade_m'] < best:
                best, best_epoch = summary['ade_m'], epoch
                save_checkpoint(out / 'best.pt', {'model_state_dict': model.state_dict(), 'config': config,
                                                  'bins': bins.tolist(), 'epoch': epoch, 'val': summary,
                                                  'best': best, 'best_epoch': best_epoch})
            print(json.dumps({'epoch': epoch, 'val_ade_m': summary['ade_m'],
                              'best_ade_m': best, 'best_epoch': best_epoch}), flush=True)
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        save_checkpoint(out / 'last.pt', {
            'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'config': config, 'bins': bins.tolist(), 'epoch': epoch,
            'best': best, 'best_epoch': best_epoch, 'elapsed_s': time.time() - start,
            **rng_payload(device, generator)})
        if is_check and epoch - best_epoch >= args.patience:
            print(f'Early stopping at {epoch}; best epoch {best_epoch}', flush=True)
            break
    checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    summary, rows = evaluate(model, test_loader, bins, device, args.viterbi_penalty,
                              args.max_jump, description='Final test (best validation checkpoint)')
    summary.update(best_epoch=best_epoch, best_val_ade_m=best)
    (out / 'test_metrics.json').write_text(json.dumps(summary, indent=2))
    with (out / 'test_per_clip.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Run saved to {out}', flush=True)


if __name__ == '__main__':
    main()
