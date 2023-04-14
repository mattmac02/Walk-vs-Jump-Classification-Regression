# **Phyphox Accelerometer Classifier**
This project is a desktop application that uses data collected from a smartphone's accelerometer to distinguish between 'walking' and 'jumping'. The application takes in accelerometer data in CSV format, and outputs a CSV file containing the labels ('walking' or 'jumping') for the corresponding input data. The system uses logistic regression as a simple classifier for this classification task.

## Getting Started
### Prerequisites
To run this project, you need to have the following installed:

- Python 3
- pandas
- scikit-learn
### Installation
Clone the repository to your local machine.
Install the required dependencies using pip.
python
Copy code
pip install pandas scikit-learn
Run the application by executing the main.py file.
### Usage
1. Prepare your accelerometer data in CSV format. The CSV file should have the following columns:

    - x: The x-axis accelerometer data.
    - y: The y-axis accelerometer data.
    - z: The z-axis accelerometer data.
2. Run the main.py file. The application will prompt you to enter your accelerometer data CSV file.

3. The application will automatically apply logistic regression to classify each data point as either 'walking' or 'jumping'.
