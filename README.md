# Time Series Elevator Failure Prediction using IoT Sensor Data

## Overview
This repository focuses on predicting elevator failures based on IoT sensor data. Predicting failures in real-time is critical to prevent unplanned downtime and maintenance costs. The dataset used for this project contains a variety of sensor readings collected over time from operational elevators. These readings are indicative of the elevator’s condition and help identify potential failures.

The core goal of this analysis is to leverage machine learning techniques to predict the operational status of the elevators— whether they are functioning normally, broken, or recovering from a failure. The dataset consists of readings from several sensors, which are processed and engineered to build robust features for classification.

## Dataset
The dataset contains multiple sensor readings, along with timestamps, capturing the real-time status of the elevator. The following columns are included:

### Data Dictionary
| Column Name     | Description                                             |
|-----------------|---------------------------------------------------------|
| `temperature`   | The temperature reading in the elevator shaft or machinery. |
| `humidity`      | The humidity level within the elevator shaft.            |
| `rpm`           | Rotational speed of elevator machinery, measured in RPM. |
| `vibrations`    | Measurement of vibrations in the elevator system.        |
| `pressure`      | Air pressure within the elevator system.                 |
| `sensor1`       | Proprietary sensor data—possibly related to vibrations.  |
| `sensor2`       | Proprietary sensor data—possibly related to pressure.    |
| `sensor3`       | Proprietary sensor data—specific to elevator operations. |
| `sensor4`       | Proprietary sensor data—linked to mechanical conditions. |
| `sensor5`       | Proprietary sensor data—additional metric for operations.|
| `sensor6`       | Proprietary sensor data—possibly related to RPM.         |
| `status`        | The current status of the elevator. It has three classes: `0` (Normal), `1` (Broken), `2` (Recovering). |

### Challenges Faced
1. **Class Imbalance**: One of the most significant challenges encountered was the extreme imbalance in the dataset. A large portion of the data corresponds to normal elevator operations (`status = 0`), while the broken and recovering states (`status = 1` and `2`) are very much underrepresented. This imbalance poses a challenge for machine learning algorithms as they tend to favor the majority class, leading to biased models that struggle to predict rare but critical events, such as breakdowns.

2. **Overfitting**: During model training, it became evident that overfitting was a challenge, especially when applying techniques like oversampling to address the class imbalance. While these techniques helped balance the data, they often led to models that performed exceptionally well on training data but struggled with generalization. Cross-validation was used to detect overfitting, and feature engineering was carefully designed to mitigate its impact.

3. **Outlier Detection**: The IoT sensor data showed signs of sensor malfunctions, leading to extreme and unrealistic readings. Outlier detection methods were implemented to identify these anomalies and replace them with NaN values for later imputation. This process was necessary to ensure that the model trained on accurate and clean data.

4. **Feature Engineering**: Transforming the raw sensor data into meaningful features played a crucial role in improving model performance. Features such as lagged variables and ratios (e.g., temperature-to-humidity ratio) were engineered to capture the temporal and relational dynamics of the sensors, enhancing the model's ability to make accurate predictions.

## Model Development
The dataset was processed using a variety of machine learning models, including Random Forest and XGBoost. These models were chosen due to their robustness and ability to handle both structured and imbalanced data. Various techniques, including random oversampling, were applied to manage the class imbalance and improve model performance.

After extensive evaluation, both models showed excellent predictive accuracy on the normal and recovering classes, but challenges persisted in predicting rare breakdowns. Techniques like cross-validation, outlier removal, and class balancing helped mitigate some of the challenges but also introduced overfitting, particularly when applied to an imbalanced dataset.

### Key Steps:
- **Data Preprocessing**: Missing values were handled using linear interpolation, and new features were engineered to capture meaningful relationships between the sensors.
- **Outlier Detection**: The IQR (Interquartile Range) and Z-score methods were used to detect outliers, which were replaced with NaN values before being imputed.
- **Cross-Validation**: To prevent overfitting, cross-validation was used to evaluate model performance on unseen data.
- **Oversampling**: Random oversampling was applied to handle class imbalance, but it led to overfitting in some cases.

## Conclusion
While the model demonstrated high accuracy for predicting normal and recovering elevator states, challenges persisted in predicting rare breakdown events due to extreme class imbalance. Future improvements may include gathering more data on breakdowns, since the dataset was extremely imbalanced. 

