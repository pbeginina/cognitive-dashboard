# Cognitive Dashboard

An interactive Streamlit application for predicting cognitive metrics based on lifestyle parameters.

## Project Description

The project started with the analysis of a real dataset containing information about sleep, caffeine intake, physical activity, stress levels, and cognitive test results. During the exploratory data analysis stage, the following aspects were studied:

- Correlations between sleep hours, caffeine, activity, and cognitive metrics
- Outliers and anomalies in the data
- Hypotheses about the influence of sleep, caffeine, and activity on memory and reaction
- The effect of body weight on sleep quality and cognitive performance
- Multifactor segmentation by gender, age, activity, and stress

After that, an interactive dashboard was created where the user can input their parameters and receive:

- Predictions of cognitive metrics (attention, memory, reaction time)
- Personalized recommendations for improvement

The project is implemented in Python using the Streamlit library and Ridge Regression models.

## User Input in the Application

- Age, gender, weight, and height (for BMI calculation)
- Number of sleep hours per day
- Number of cups of coffee per day
- Level of physical activity (in hours per week)

## Output Metrics

- Memory accuracy (N-Back Accuracy)
- Reaction time (PVT and Stroop)
- Level of daytime sleepiness

## Recommendations

Based on data analysis and identified patterns, the application provides suggestions for improving sleep, nutrition, and physical activity.

## How to Run

1. Clone the repository:
   
   git clone https://github.com/your_username/cognitive-dashboard.git
   cd cognitive-dashboard

2. Install dependencies:
   
  pip install -r requirements.txt

3. Run the application:

   streamlit run app.py


## Technologies Used

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn (Ridge Regression)
