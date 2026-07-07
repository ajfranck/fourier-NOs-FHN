import torch
import numpy as np
import tempfile
import sys
import os
from pathlib import Path

sys.path.append('.')

from fhn_fno.config import Config, DataConfig, FHNParams
from fhn_fno.data.generate_fhn import FDBackend, sample_initial_conditions, generate_dataset
from fhn_fno.data.dataset import FHNOperatorDataset
from fhn_fno.models.fno import FNO, SpectralConv1d
from fhn_fno.eval.metrics import relative_l2_error


def create_test_file():
    # local dir instead of tempdir, which is flaky on Windows
    test_dir = Path("test_temp")
    test_dir.mkdir(exist_ok=True)
    return test_dir / "test_data.h5"


def test_dataset_shapes():
    config = DataConfig(nx=64, T=1.0, n_timesteps=10)
    
    test_file = create_test_file()
    try:
        generate_dataset(config, n_samples=4, output_file=str(test_file), verbose=False)

        dataset = FHNOperatorDataset(str(test_file), mode="single_step", train=True)

        assert len(dataset) > 0
        sample = dataset[0]
        assert sample['input'].shape == (2, 64)  # (channels, nx)
        assert sample['target'].shape == (2, 64)
        assert sample['params'].shape == (5,)  # 5 FHN parameters

    finally:
        if test_file.exists():
            test_file.unlink()
        if test_file.parent.exists() and not any(test_file.parent.iterdir()):
            test_file.parent.rmdir()


def test_fno_forward():
    model = FNO(modes=8, width=32, n_layers=2, dim=1)

    batch_size = 4
    nx = 128
    x = torch.randn(batch_size, 2, nx)

    y = model(x)

    assert y.shape == (batch_size, 2, nx)
    assert not torch.isnan(y).any()


def test_spectral_conv():
    layer = SpectralConv1d(in_channels=2, out_channels=2, modes=8)
    
    x = torch.randn(4, 2, 64)
    y = layer(x)
    
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_training_step():
    config = DataConfig(nx=64, T=1.0, n_timesteps=5)
    
    test_file = create_test_file()
    try:
        generate_dataset(config, n_samples=8, output_file=str(test_file), verbose=False)
        
        dataset = FHNOperatorDataset(str(test_file), mode="single_step", train=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        model = FNO(modes=8, width=32, n_layers=2, dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        model.train()
        for batch in loader:
            x = batch['input']
            y = batch['target']

            pred = model(x)
            loss = criterion(pred, y)

            assert not torch.isnan(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for p in model.parameters():
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any()

            break  # one batch is enough

    finally:
        if test_file.exists():
            test_file.unlink()
        if test_file.parent.exists() and not any(test_file.parent.iterdir()):
            test_file.parent.rmdir()


def test_fd_solver():
    solver = FDBackend(nx=32, dx=1.0/32)
    params = FHNParams()
    
    u0, v0 = sample_initial_conditions(32, None, "grf", seed=42)

    sol = solver.solve(u0, v0, params, T=0.1, dt=0.001, n_save=5)

    assert 'u' in sol
    assert 'v' in sol
    assert 't' in sol

    assert sol['u'].shape[0] == 6  # n_save + 1
    assert sol['u'].shape[1] == 32  # nx

    assert not np.isnan(sol['u']).any()
    assert not np.isnan(sol['v']).any()


def test_rollout():
    model = FNO(modes=8, width=32, n_layers=2, dim=1)
    
    x0 = torch.randn(2, 2, 64)  # batch=2, channels=2, nx=64

    trajectory = model.rollout(x0, n_steps=5)

    # (batch, n_steps+1, channels, nx)
    assert trajectory.shape == (2, 6, 2, 64)
    assert not torch.isnan(trajectory).any()


def test_relative_l2_error():
    pred = torch.randn(4, 2, 64)
    target = torch.randn(4, 2, 64)

    error = relative_l2_error(pred, target)

    assert error.shape == ()
    assert error > 0
    assert not torch.isnan(error)
    
    error_perfect = relative_l2_error(target, target)
    assert error_perfect < 1e-6


def run_all_tests():
    # flip to False to skip a test
    RUN_TESTS = {
        "Dataset shapes": True,
        "FNO forward": True,
        "Spectral conv": True,
        "Training step": True,
        "FD solver": True,
        "Rollout": True,
        "Relative L2 error": True,
    }
    
    tests = [
        ("Dataset shapes", test_dataset_shapes),
        ("FNO forward", test_fno_forward),
        ("Spectral conv", test_spectral_conv),
        ("Training step", test_training_step),
        ("FD solver", test_fd_solver),
        ("Rollout", test_rollout),
        ("Relative L2 error", test_relative_l2_error),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        if not RUN_TESTS.get(test_name, True):
            print(f"Skipping {test_name}...")
            skipped += 1
            continue
            
        try:
            print(f"Running {test_name}...", end=" ")
            test_func()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("All tests passed, ballin!")
        return True
    else:
        print("Some tests failed, not cool")
        return False


def main():
    print("Running FHN-FNO Smoke Tests")
    print("=" * 40)
    success = run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()