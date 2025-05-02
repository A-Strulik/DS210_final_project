mod Read_Clean;
use Read_Clean::{process_csv_file, convert_to_xy};
use linfa::Dataset;
use linfa::traits::Fit;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2, array, s};
use smartcore::model_selection::train_test_split;
use linfa::prelude::Predict;

// Calculate the R squared score for predictions vs actual values
fn calculate_r2_score(predictions: &Array1<f64>, actual: &Array1<f64>) -> f64 {
    let mean_actual = actual.mean().unwrap();
    let total_variance = actual.iter()
        .map(|&value| (value - mean_actual).powi(2))
        .sum::<f64>();
    let residual_sum_of_squares = predictions.iter()
        .zip(actual.iter())
        .map(|(pred, act)| (act - pred).powi(2))
        .sum::<f64>();

    1.0 - (residual_sum_of_squares / total_variance)
}

// Calculate Root Mean Squared Error (RMSE) for predictions vs actual values
fn calculate_rmse(predictions: &Array1<f64>, actual: &Array1<f64>) -> f64 {
    let sum_squared_error = predictions.iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>();
    (sum_squared_error / actual.len() as f64).sqrt()
}

fn main() {
    // Read and clean the data
    let filtered_records = process_csv_file();
    
    // Define the standard deviation of the noise 
    let noise_stddev = 0.025 * filtered_records.iter().map(|r| r.WorldwideGross).max().unwrap() as f64;

    let (x, y) = convert_to_xy(filtered_records, noise_stddev);

    // Split the data into training and testing sets (80% train, 20% test)
    let (x_train, x_test, y_train, y_test) = train_test_split(
        &x, 
        &y, 
        0.2, 
        true,
    );

    // Create the training dataset
    let train_dataset = Dataset::new(x_train.clone(), y_train.clone());
    let test_dataset = Dataset::new(x_test.clone(), y_test.clone());

    // Fit the linear regression model
    let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&train_dataset).unwrap();

    // Print the coefficients and intercept
    println!("Coefficients: {:?}", model.params());
    println!("Intercept: {:?}", model.intercept());

    // Make predictions using the fitted model
    let predictions = model.predict(&x_test);

    // Evaluate the model using rmse and R squared
    let rmse = calculate_rmse(&predictions, &y_test);
    println!("Root Mean Squared Error (RMSE): {}", rmse);

    let r2_score = calculate_r2_score(&predictions, &y_test);
    println!("R^2 Score: {}", r2_score);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_calculate_r2_score_perfect_fit() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let r2 = calculate_r2_score(&y_pred, &y_true);
        assert!((r2 - 1.0).abs() < 1e-10); // Perfect fit
    }

    #[test]
    fn test_calculate_r2_score_zero_fit() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![2.0, 2.0, 2.0];
        let r2 = calculate_r2_score(&y_pred, &y_true);
        assert!(r2 < 1.0); // Poor fit
    }
}