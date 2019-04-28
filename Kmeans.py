import argparse
from model import Kmeans
from utils import *

def parse_args():
	parser = argparse.ArgumentParser(description='Kmeans clustering on two different dataset.')
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--K', required=True, type=int)
	parser.add_argument('--init', type=str)
	parser.add_argument('--maxiter', type=int)
	parser.add_argument('--seed', type=int)
	return parser.parse_args()

def print_SSE(sse):
	print("Kmeans Sum of Square Error of each epoch:")
	for i in range(len(sse)):
		print("epoch %3d: %8.3f   "%(i, sse[i]), end='')
		if (i+1)%3 == 0 or i == len(sse)-1:
			print()
	print("Kmeans Final Sum of Square Error:")
	print(sse[-1])

def main(args):

	if args.dataset == 'abalone':
		X, Y = load_abalone()
	else:
		X, Y = load_iris()
	
	if args.init == 'kmeans++':
		init = 'kmeans++'
	else:
		init = ''
	
	if args.maxiter == None:
		maxiter = 20
	else:
		maxiter = args.maxiter

	kmeans = Kmeans(K=args.K, max_iter=maxiter, init=init, seed=args.seed)
	kmeans.fit(X)
	print_SSE(kmeans.SSE)
	
	
if __name__ == '__main__':
	args = parse_args()
	main(args)
	