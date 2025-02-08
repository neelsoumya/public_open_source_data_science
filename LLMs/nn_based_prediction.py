import meow
import pandas as pd

class nn_based_prediction:

    def predict(self, input_dataset: pd.DataFrame):

        # Create instance of class.
        ml = meow.ML()

        # Set prediction target.
        target_column_name = "eta_is"

        # Try to obtain ML prediction.
        prediction_set = ml.predict(input_dataset, target_column_name)

        # Return predictions.
        return prediction_set