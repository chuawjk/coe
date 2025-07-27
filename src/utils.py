import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class TemporalCrossValidator:
    def __init__(self, initial_train_years=5, test_period_months=6, step_months=6):
        """
        Temporal cross-validation for time series data
        
        Args:
            initial_train_years: Initial training period in years
            test_period_months: Length of each test period in months  
            step_months: Step size between folds in months
        """
        self.initial_train_years = initial_train_years
        self.test_period_months = test_period_months
        self.step_months = step_months
    
    def get_time_splits(self, df, date_col='ds'):
        """
        Generate time-based train/test splits with CONSTANT training window size
        
        Uses sliding window approach:
        - Training window size stays constant across folds
        - As we move forward, drop oldest data and add newest data
        
        Returns list of (train_df, test_df) tuples
        """
        # Sort by date to ensure proper ordering
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        
        # Get date range
        start_date = df_sorted[date_col].min()
        end_date = df_sorted[date_col].max()
        
        # Calculate training window size in months
        train_window_months = self.initial_train_years * 12
        
        splits = []
        current_test_start = start_date + pd.DateOffset(years=self.initial_train_years)
        
        while current_test_start < end_date:
            # Define test period
            test_start = current_test_start
            test_end = current_test_start + pd.DateOffset(months=self.test_period_months)
            
            # Stop if test period extends beyond available data
            if test_end > end_date:
                break
            
            # Define training period with FIXED window size
            # Training ends right before test starts
            train_end = test_start
            train_start = train_end - pd.DateOffset(months=train_window_months)
            
            # Create train/test splits
            train_mask = ((df_sorted[date_col] >= train_start) & 
                         (df_sorted[date_col] < train_end))
            test_mask = ((df_sorted[date_col] >= test_start) & 
                        (df_sorted[date_col] < test_end))
            
            train_split = df_sorted[train_mask].copy()
            test_split = df_sorted[test_mask].copy()
            
            # Only include splits with sufficient data
            if len(train_split) > 100 and len(test_split) > 10:
                splits.append({
                    'fold': len(splits) + 1,
                    'train_start': train_split[date_col].min(),
                    'train_end': train_split[date_col].max(), 
                    'test_start': test_split[date_col].min(),
                    'test_end': test_split[date_col].max(),
                    'train_df': train_split,
                    'test_df': test_split,
                    'train_size': len(train_split),
                    'test_size': len(test_split),
                    'train_window_start': train_start,
                    'train_window_end': train_end
                })
            
            # Move to next fold
            current_test_start += pd.DateOffset(months=self.step_months)
        
        return splits
    
    def evaluate_model(self, model_func, splits, verbose=False):
        """
        Evaluate a model using temporal cross validation
        
        Args:
            model_func: Function that takes (train_df, test_df) and returns predictions
            splits: List of train/test splits from get_time_splits()
            verbose: Whether to print progress
            
        Returns:
            Dictionary with CV results
        """
        fold_results = []
        
        for split in splits:
            fold_num = split['fold']
            train_df = split['train_df']
            test_df = split['test_df']
            
            if verbose:
                print(f"\\nFold {fold_num}:")
                print(f"  Train: {split['train_start'].date()} to {split['train_end'].date()} ({split['train_size']} samples)")
                print(f"  Test:  {split['test_start'].date()} to {split['test_end'].date()} ({split['test_size']} samples)")
            
            try:
                # Get predictions from model
                predictions = model_func(train_df, test_df)
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(test_df['y'], predictions['yhat']))
                
                # Calculate RMSE by vehicle class
                vehicle_class_rmses = {}
                vehicle_classes = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
                
                for vehicle_class in vehicle_classes:
                    mask = test_df[f'vehicle_class_{vehicle_class}'] == True
                    if mask.any():
                        class_rmse = np.sqrt(mean_squared_error(
                            test_df.loc[mask, 'y'], 
                            predictions.loc[mask, 'yhat']
                        ))
                        vehicle_class_rmses[vehicle_class] = class_rmse
                
                fold_result = {
                    'fold': fold_num,
                    'rmse': rmse,
                    'vehicle_class_rmses': vehicle_class_rmses,
                    'train_size': split['train_size'],
                    'test_size': split['test_size'],
                    'test_period': f"{split['test_start'].date()} to {split['test_end'].date()}"
                }
                
                fold_results.append(fold_result)
                
                if verbose:
                    print(f"  Overall RMSE: {rmse:.2f}")
                    for vc, vc_rmse in vehicle_class_rmses.items():
                        print(f"  {vc} RMSE: {vc_rmse:.2f}")
                        
            except Exception as e:
                print(f"  Error in fold {fold_num}: {str(e)}")
                continue
        
        # Calculate summary statistics
        if fold_results:
            rmses = [r['rmse'] for r in fold_results]
            mean_rmse = np.mean(rmses)
            std_rmse = np.std(rmses)
            
            # Average vehicle class RMSEs
            avg_vehicle_rmses = {}
            for vehicle_class in vehicle_classes:
                class_rmses = [r['vehicle_class_rmses'].get(vehicle_class) 
                              for r in fold_results 
                              if vehicle_class in r['vehicle_class_rmses']]
                if class_rmses:
                    avg_vehicle_rmses[vehicle_class] = {
                        'mean': np.mean(class_rmses),
                        'std': np.std(class_rmses),
                        'count': len(class_rmses)
                    }
            
            summary = {
                'mean_rmse': mean_rmse,
                'std_rmse': std_rmse,
                'min_rmse': min(rmses),
                'max_rmse': max(rmses),
                'num_folds': len(fold_results),
                'avg_vehicle_rmses': avg_vehicle_rmses,
                'fold_results': fold_results
            }
            
            if verbose:
                print(f"\\n{'='*50}")
                print(f"TEMPORAL CROSS VALIDATION SUMMARY")
                print(f"{'='*50}")
                print(f"Number of folds: {summary['num_folds']}")
                print(f"Average RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
                print(f"RMSE range: [{min(rmses):.2f}, {max(rmses):.2f}]")
                print(f"\\nVehicle Class Average RMSEs:")
                for vc, stats in avg_vehicle_rmses.items():
                    print(f"  {vc}: {stats['mean']:.2f} ± {stats['std']:.2f} (n={stats['count']})")
            
            return summary
        else:
            print("No successful folds completed!")
            return None
