"""Test the CLI of the aitw-infiltration-surrogate package."""
import os

import pandas as pd
from click.testing import CliRunner


def test_cli_existing_output_dir(dataset_file_path, tmpdir):
    """Test the CLI train command with an existing output directory."""
    from aitw_infiltration_surrogate.cli import cli

    tmp_output_dir = tmpdir / "output"
    os.makedirs(tmp_output_dir)

    runner = CliRunner()
    result = runner.invoke(cli, [
        'train',
        '--n_epochs', '1',
        '--data_file', dataset_file_path,
        '--output_dir', str(tmp_output_dir),
    ])

    assert result.exit_code == 1
    assert 'already exists' in result.output


def test_cli_train_and_inference(dataset_file_path, tmpdir):
    """Test the CLI train command."""
    from aitw_infiltration_surrogate.cli import cli

    tmp_output_dir = tmpdir / "output"

    runner = CliRunner()
    output_weights_file = os.path.join(tmp_output_dir, "weights.pt")
    result = runner.invoke(cli, [
        'train',
        '--n_epochs', '1',
        '--data_file', dataset_file_path,
        '--output_dir', str(tmp_output_dir),
    ])

    assert result.exit_code == 0
    assert os.path.exists(output_weights_file)

    inference_results = os.path.join(tmpdir, 'inference_results.csv')
    result = runner.invoke(cli, [
        'inference',
        '--input_file', dataset_file_path,
        '--weight_file', output_weights_file,
        '--output_file', inference_results,
    ])

    assert result.exit_code == 0
    assert os.path.exists(inference_results)

    inference_results_df = pd.read_csv(inference_results)
    assert 'sat_time' in inference_results_df.columns
    assert 'expected_sat_time' in inference_results_df.columns

def test_cli_inference_with_default_weights(dataset_file_path, tmpdir):
    """Test the CLI inference command with default weights."""
    from aitw_infiltration_surrogate.cli import cli

    runner = CliRunner()
    inference_results = os.path.join(tmpdir, 'inference_results.csv')
    result = runner.invoke(cli, [
        'inference',
        '--input_file', dataset_file_path,
        '--output_file', inference_results,
    ])

    assert result.exit_code == 0
    assert os.path.exists(inference_results)

    inference_results_df = pd.read_csv(inference_results)
    assert 'sat_time' in inference_results_df.columns
    assert 'expected_sat_time' in inference_results_df.columns


def test_cli_train_nst(dataset_nst_path, tmpdir):
    """Test the CLI train command with a dataset without `sat_time` column."""
    from aitw_infiltration_surrogate.cli import cli

    runner = CliRunner()
    inference_results = os.path.join(tmpdir, 'inference_results.csv')
    result = runner.invoke(cli, [
        'inference',
        '--input_file', dataset_nst_path,
        '--output_file', inference_results,
    ])

    assert result.exit_code == 0
    assert os.path.exists(inference_results)

    inference_results_df = pd.read_csv(inference_results)
    assert 'sat_time' in inference_results_df.columns
    assert 'expected_sat_time' not in inference_results_df.columns


def test_cli_train_wrong_values(caplog, dataset_wnv_path, tmpdir):
    """Test the CLI train command with wrong negative values."""
    from aitw_infiltration_surrogate.cli import cli

    tmp_output_dir = tmpdir / "output"

    runner = CliRunner()
    with caplog.at_level("ERROR"):
        result = runner.invoke(cli, [
            'train',
            '--n_epochs', '1',
            '--data_file', dataset_wnv_path,
            '--output_dir', str(tmp_output_dir),
        ])

    assert result.exit_code != 0
    assert "Data contains non-positive values" in caplog.text

def test_cli_train_wrong_shape(caplog, dataset_ws_path, tmpdir):
    """Test the CLI train command with wrong shape."""
    from aitw_infiltration_surrogate.cli import cli

    tmp_output_dir = tmpdir / "output"

    runner = CliRunner()
    with caplog.at_level("ERROR"):
        result = runner.invoke(cli, [
            'train',
            '--n_epochs', '1',
            '--data_file', dataset_ws_path,
            '--output_dir', str(tmp_output_dir),
        ])

    assert result.exit_code != 0
    assert "Data shape" in caplog.text

def test_cli_inference_wrong_values(caplog, dataset_wnv_path, tmpdir):
    """Test the CLI inference command with wrong negative values."""
    from aitw_infiltration_surrogate.cli import cli

    runner = CliRunner()
    with caplog.at_level("ERROR"):
        result = runner.invoke(cli, [
            'inference',
            '--input_file', dataset_wnv_path,
            '--output_file', str(tmpdir / 'inference_results.csv'),
        ])

    assert result.exit_code != 0
    assert "Input data contains non-positive values" in caplog.text

def test_cli_inference_wrong_shape(caplog, dataset_ws_path, tmpdir):
    """Test the CLI inference command with wrong shape."""
    from aitw_infiltration_surrogate.cli import cli

    runner = CliRunner()
    with caplog.at_level("ERROR"):
        result = runner.invoke(cli, [
            'inference',
            '--input_file', dataset_ws_path,
            '--output_file', str(tmpdir / 'inference_results.csv'),
        ])

    assert result.exit_code != 0
    assert "Input data shape" in caplog.text
