#!/usr/bin/env python3
"""
Main experiment runner for GS-RVFL.
Runs experiments on all datasets and saves results.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gs_rvfl import GSRVFLClassifier, GSRVFLRegressor
from gs_rvfl.operators import (
    create_motion_operators,
    create_sensor_operators,
    create_tabular_operators
)
from gs_rvfl.utils import accuracy_score, f1_score, create_train_test_split

from experiments.data.loader import DatasetLoader
from experiments.data.datasets import (
    TABULAR_DATASETS,
    SENSOR_DATASETS,
    VIDEO_DATASETS,
    SKELETON_DATASETS,
    DOMAIN_SPECIFIC_DATASETS,
    ALGORITHMS
)

warnings.filterwarnings('ignore')


class ExperimentRunner:
    """
    Runner for GS-RVFL experiments.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        output_dir: str = 'results'
    ):
        self.random_state = random_state
        self.output_dir = output_dir
        self.dataset_loader = DatasetLoader(random_state=random_state)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    
    def get_operators(self, domain: str) -> Tuple[List, List]:
        """Get operators for a given domain."""
        if domain == 'tabular':
            return create_tabular_operators()
        elif domain == 'sensor':
            return create_sensor_operators(time_window=1)
        elif domain == 'video':
            return create_motion_operators(time_window=1)
        elif domain == 'skeleton':
            return create_motion_operators(time_window=1, include_acceleration=True)
        else:
            return [], []
    
    def run_single_experiment(
        self,
        dataset_name: str,
        domain: str,
        n_hidden: int = 100,
        lambda_reg: float = 1e-4,
        scale: float = 1.0,
        activation: str = 'tanh',
        n_runs: int = 10,
        test_size: float = 0.2
    ) -> pd.DataFrame:
        """
        Run a single experiment on a dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        domain : str
            Domain of the dataset.
        n_hidden : int
            Number of hidden nodes.
        lambda_reg : float
            Regularization parameter.
        scale : float
            Weight scale.
        activation : str
            Activation function.
        n_runs : int
            Number of runs.
        test_size : float
            Test set proportion.
        
        Returns
        -------
        results_df : pd.DataFrame
            Results dataframe.
        """
        print(f"\n{'='*60}")
        print(f"Running experiment on {dataset_name} ({domain})")
        print(f"{'='*60}")
        
        # Load dataset
        X, y = self.dataset_loader.load_dataset(dataset_name)
        
        results = []
        
        for run in range(n_runs):
            run_seed = self.random_state + run
            
            # Split data
            X_train, X_test, y_train, y_test = create_train_test_split(
                X, y, test_size=test_size, random_state=run_seed
            )
            
            # Standardize
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Get operators
            g_ops, b_ops = self.get_operators(domain)
            
            # Create model
            model = GSRVFLClassifier(
                n_hidden=n_hidden,
                lambda_reg=lambda_reg,
                scale=scale,
                activation=activation,
                random_state=run_seed,
                g_operators=g_ops,
                b_operators=b_ops
            )
            
            # Train and evaluate
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            start_time = time.time()
            y_pred = model.predict(X_test_scaled)
            test_time = time.time() - start_time
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Get model info
            n_g = model.n_g_features
            n_b = model.n_b_features
            n_total = model.n_total_features
            
            results.append({
                'run': run,
                'dataset': dataset_name,
                'domain': domain,
                'accuracy': acc,
                'f1_score': f1,
                'train_time': train_time,
                'test_time': test_time,
                'n_hidden': n_hidden,
                'n_g_features': n_g,
                'n_b_features': n_b,
                'n_total_features': n_total,
                'lambda_reg': lambda_reg,
                'scale': scale,
                'activation': activation
            })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = os.path.join(
            self.output_dir, 'raw',
            f'{dataset_name}_{domain}_results.csv'
        )
        results_df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\nResults for {dataset_name}:")
        print(f"  Mean accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
        print(f"  Mean F1: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
        print(f"  Saved to: {output_file}")
        
        return results_df
    
    def run_all_experiments(
        self,
        n_hidden: int = 100,
        lambda_reg: float = 1e-4,
        scale: float = 1.0,
        activation: str = 'tanh',
        n_runs: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Run experiments on all datasets.
        """
        all_results = {}
        
        # Tabular datasets (reduction validation)
        print("\n" + "="*80)
        print("RUNNING TABULAR DATASETS (REDUCTION VALIDATION)")
        print("="*80)
        
        for dataset_name in list(TABULAR_DATASETS.keys())[:3]:  # Reduced for demo
            results = self.run_single_experiment(
                dataset_name, 'tabular',
                n_hidden=n_hidden,
                lambda_reg=lambda_reg,
                scale=scale,
                activation=activation,
                n_runs=n_runs
            )
            all_results[f'tabular_{dataset_name}'] = results
        
        # Sensor datasets
        print("\n" + "="*80)
        print("RUNNING SENSOR DATASETS")
        print("="*80)
        
        for dataset_name in list(SENSOR_DATASETS.keys()):
            results = self.run_single_experiment(
                dataset_name, 'sensor',
                n_hidden=n_hidden,
                lambda_reg=lambda_reg,
                scale=scale,
                activation=activation,
                n_runs=n_runs
            )
            all_results[f'sensor_{dataset_name}'] = results
        
        # Video datasets
        print("\n" + "="*80)
        print("RUNNING VIDEO DATASETS")
        print("="*80)
        
        for dataset_name in list(VIDEO_DATASETS.keys())[:3]:  # Reduced for demo
            results = self.run_single_experiment(
                dataset_name, 'video',
                n_hidden=n_hidden,
                lambda_reg=lambda_reg,
                scale=scale,
                activation=activation,
                n_runs=n_runs
            )
            all_results[f'video_{dataset_name}'] = results
        
        # Skeleton datasets
        print("\n" + "="*80)
        print("RUNNING SKELETON DATASETS")
        print("="*80)
        
        for dataset_name in list(SKELETON_DATASETS.keys()):
            results = self.run_single_experiment(
                dataset_name, 'skeleton',
                n_hidden=n_hidden,
                lambda_reg=lambda_reg,
                scale=scale,
                activation=activation,
                n_runs=n_runs
            )
            all_results[f'skeleton_{dataset_name}'] = results
        
        # Combine all results
        all_df = pd.concat(all_results.values(), ignore_index=True)
        combined_file = os.path.join(self.output_dir, 'processed', 'all_results.csv')
        all_df.to_csv(combined_file, index=False)
        print(f"\nAll results saved to: {combined_file}")
        
        return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run GS-RVFL experiments')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--domain', type=str, default='tabular', 
                       choices=['tabular', 'sensor', 'video', 'skeleton'])
    parser.add_argument('--n_runs', type=int, default=10, help='Number of runs')
    parser.add_argument('--n_hidden', type=int, default=100, help='Number of hidden nodes')
    parser.add_argument('--lambda_reg', type=float, default=1e-4, help='Regularization')
    parser.add_argument('--scale', type=float, default=1.0, help='Weight scale')
    parser.add_argument('--activation', type=str, default='tanh', 
                       choices=['sigmoid', 'tanh', 'relu', 'sine', 'radbas'])
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    if args.all:
        runner.run_all_experiments(
            n_hidden=args.n_hidden,
            lambda_reg=args.lambda_reg,
            scale=args.scale,
            activation=args.activation,
            n_runs=args.n_runs
        )
    elif args.dataset:
        runner.run_single_experiment(
            args.dataset,
            args.domain,
            n_hidden=args.n_hidden,
            lambda_reg=args.lambda_reg,
            scale=args.scale,
            activation=args.activation,
            n_runs=args.n_runs
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()