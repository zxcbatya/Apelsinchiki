@echo off
echo Ride Pricing Optimization Analysis
echo =================================

echo Installing required packages...
pip install -r requirements.txt

echo Running data exploration...
python data_exploration.py

echo Generating data visualizations...
python visualize_data.py

echo Training the neural network model...
python model.py

echo Making predictions on new orders...
python predict_new_orders.py

echo Performing comprehensive model evaluation...
python model_evaluation.py

echo.
echo Analysis complete!
echo You can also open pricing_model.ipynb in Jupyter Notebook for interactive analysis.
pause