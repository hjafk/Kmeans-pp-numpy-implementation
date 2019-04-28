from sklearn import datasets
import numpy as np
import pandas as pd
from math import exp

def load_abalone():
	col_names = ["sex", "length", "diameter", "height", "whole weight",
             "shucked weight", "viscera weight", "shell weight", "rings"]
	df = pd.read_csv('abalone.data', names=col_names)

	df_sex = pd.get_dummies(df['sex'], prefix='sex')
	Y_abalone = df['rings'].values
	df = df.drop(['sex', 'rings'], axis=1)
	df = pd.concat([df, df_sex], axis=1)
	X_abalone = df.values
	return X_abalone, Y_abalone
	

def load_iris():
	iris = datasets.load_iris()
	X = iris.data
	Y = iris.target
	return X, Y
	