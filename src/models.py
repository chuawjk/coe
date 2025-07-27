from prophet import Prophet
import pandas as pd

class BaseModel:
    def __init__(
        self,
        vehicle_class: str,
        regressors: list[str],
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        regressor_prior_scale: float = 10.0
    ):
        self.vehicle_class = vehicle_class
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
        )
        for regressor in regressors:
            self.model.add_regressor(regressor, mode='multiplicative', prior_scale=regressor_prior_scale)

    def filter_training_data(self, data: pd.DataFrame):
        """
        Prepare training data by filtering for the vehicle class
        """
        data = data[data[f'vehicle_class_{self.vehicle_class}'] == True]
        return data

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the model to the training data
        """
        filtered_data = self.filter_training_data(train_df)
        self.model.fit(filtered_data)

    def predict(self, test_df: pd.DataFrame):
        return self.model.predict(test_df)


class EnsembleModel:
    def __init__(
        self,
        vehicle_classes: list[str],
        regressors: list[str],
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        regressor_prior_scale: float = 10.0
        ):
        self.vehicle_classes = vehicle_classes
        self.regressors = regressors
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.regressor_prior_scale = regressor_prior_scale

        self.base_models = {}
        self.init_base_models()

    def init_base_models(self):
        for vehicle_class in self.vehicle_classes:
            self.base_models[vehicle_class] = BaseModel(
                vehicle_class,
                self.regressors,
                self.changepoint_prior_scale,
                self.seasonality_prior_scale,
                self.holidays_prior_scale,
                self.n_changepoints,
                self.changepoint_range,
                self.regressor_prior_scale
            )

    def fit(self, train_df: pd.DataFrame):
        # Fit base models first
        print("Fitting base models")
        for vehicle_class in self.vehicle_classes:
            self.base_models[vehicle_class].fit(train_df)

    def predict(self, df: pd.DataFrame):
        """
        Predict while preserving exact row ordering of input dataframe
        Handles both DataFrame and Series inputs
        """
        # Convert Series to DataFrame if needed
        if isinstance(df, pd.Series):
            df = df.to_frame().T  # Convert Series to single-row DataFrame
        elif not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be pandas DataFrame or Series, got {type(df)}")
        
        # Create output dataframe with same index as input
        output_df = pd.DataFrame(index=df.index)
        
        # Add ds column from input
        if 'ds' in df.columns:
            output_df['ds'] = df['ds']
        else:
            print("Warning: 'ds' column not found - predictions may not have timestamps")
        
        # Initialize prediction columns with NaN
        pred_columns = ['yhat', 'yhat_lower', 'yhat_upper']  # Common Prophet outputs
        for col in pred_columns:
            output_df[col] = float('nan')
        
        # Process each vehicle class with improved single-class handling
        assigned_rows = pd.Series(False, index=df.index)  # Track which rows have been assigned
        
        for vehicle_class in self.vehicle_classes:
            # Check if vehicle class column exists
            vehicle_class_col = f'vehicle_class_{vehicle_class}'
            if vehicle_class_col not in df.columns:
                print(f"Warning: Column '{vehicle_class_col}' not found. Available columns: {df.columns.tolist()}")
                continue
                
            # Filter rows for this vehicle class
            mask = df[vehicle_class_col] == True
            class_df = df[mask]
            
            if len(class_df) > 0:
                try:
                    # Get predictions from the base model for this class
                    class_pred = self.base_models[vehicle_class].predict(class_df)
                    
                    # Assign predictions back to original positions using .loc
                    for col in pred_columns:
                        if col in class_pred.columns:
                            output_df.loc[mask, col] = class_pred[col].values
                    
                    # Mark these rows as assigned
                    assigned_rows |= mask
                    
                except Exception as e:
                    print(f"Error predicting for {vehicle_class}: {str(e)}")
                    continue
        
        # Handle case where no rows were assigned (fallback strategy)
        unassigned_mask = ~assigned_rows
        if unassigned_mask.any():
            print(f"Warning: {unassigned_mask.sum()} rows have no vehicle class assigned!")
            
            # Fallback strategies for unassigned rows
            if len(self.vehicle_classes) == 1:
                # Single vehicle class case - assign all unassigned rows to this class
                fallback_class = self.vehicle_classes[0]
                print(f"Single vehicle class detected. Assigning unassigned rows to {fallback_class}")
                
                try:
                    unassigned_df = df[unassigned_mask]
                    fallback_pred = self.base_models[fallback_class].predict(unassigned_df)
                    
                    for col in pred_columns:
                        if col in fallback_pred.columns:
                            output_df.loc[unassigned_mask, col] = fallback_pred[col].values
                            
                except Exception as e:
                    print(f"Fallback prediction failed: {str(e)}. Filling with zeros.")
                    for col in pred_columns:
                        output_df.loc[unassigned_mask, col] = 0
            
            elif len(self.vehicle_classes) > 1:
                # Multiple vehicle classes - use most common class or average predictions
                # Find the class with most training data as fallback
                class_sizes = {}
                for vc in self.vehicle_classes:
                    vc_col = f'vehicle_class_{vc}'
                    if vc_col in df.columns:
                        class_sizes[vc] = df[vc_col].sum()
                
                if class_sizes:
                    fallback_class = max(class_sizes, key=class_sizes.get)
                    print(f"Using {fallback_class} (largest class) for unassigned rows")
                    
                    try:
                        unassigned_df = df[unassigned_mask]
                        fallback_pred = self.base_models[fallback_class].predict(unassigned_df)
                        
                        for col in pred_columns:
                            if col in fallback_pred.columns:
                                output_df.loc[unassigned_mask, col] = fallback_pred[col].values
                                
                    except Exception as e:
                        print(f"Fallback prediction failed: {str(e)}. Filling with zeros.")
                        for col in pred_columns:
                            output_df.loc[unassigned_mask, col] = 0
                else:
                    # No valid classes found - fill with zeros
                    print("No valid vehicle classes found. Filling unassigned rows with zeros.")
                    for col in pred_columns:
                        output_df.loc[unassigned_mask, col] = 0
            else:
                # No vehicle classes defined - fill with zeros
                print("No vehicle classes defined. Filling with zeros.")
                for col in pred_columns:
                    output_df.loc[unassigned_mask, col] = 0
        
        return output_df