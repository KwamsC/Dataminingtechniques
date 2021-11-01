import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
class Assignment:
	def __init__(self):
		self.try_method()

	def try_method(self):
		# save filepath to variable for easier access
		melbourne_file_path = 'melb_data.csv'
		melbourne_data = pd.read_csv(melbourne_file_path) 

		# Filter rows with missing price values
		melbourne_data = melbourne_data.dropna(axis=0)
		
		# Choose target and features
		y = melbourne_data.Price

		melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
		X = melbourne_data[melbourne_features]

		# Define model
		melbourne_model = DecisionTreeRegressor(random_state=1)
		# z = melbourne_model.fit(X, y)

		# predicted_home_prices = melbourne_model.predict(X)
		# print(mean_absolute_error(y, predicted_home_prices))

		train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
		# melbourne_model = DecisionTreeRegressor()
		
		melbourne_model.fit(train_X, train_y)

		val_predictions = melbourne_model.predict(val_X)
		# print(mean_absolute_error(val_y, val_predictions))
		# print(val_predictions.head())

		# def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
		# 	model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
		# 	model.fit(train_X, train_y)
		# 	preds_val = model.predict(val_X)
		# 	mae = mean_absolute_error(val_y, preds_val)
		# 	return(mae)
			
		# for max_leaf_nodes in [5, 50, 500, 5000]:
		# 	my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
		# 	print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

		# forest_model = RandomForestRegressor(random_state=1)
		# forest_model.fit(train_X, train_y)
		# melb_preds = forest_model.predict(val_X)
		# print(mean_absolute_error(val_y, melb_preds))

		missing_val_count_by_column = (melbourne_data.isnull().sum())
		print(missing_val_count_by_column)


	



		# print("Making predictions for the following 5 houses:")
		# print(X.head())
		# print("The predictions are")
		# print(melbourne_model.predict(X.head()))

if __name__ == '__main__':
	a = Assignment()