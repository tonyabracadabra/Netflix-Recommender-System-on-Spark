import os
import urllib
from pyspark import SparkContext, SparkConf


complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

datasets_path = os.path.join('..', 'datasets')


def init_spark_context():
    # load spark context
    conf = SparkConf().setAppName("movielens_movie_recommendation")
    # IMPORTANT: pass aditional Python modules to each worker
    sc = SparkContext(conf=conf)
 
    return sc

def preprocess_movie_lens(sc, folder, ratings_file, movies_file):
	ratings_file = os.path.join(datasets_path, folder, ratings_file)

	ratings_raw = sc.textFile(ratings_file)
	ratings_header = ratings_raw.first()

	# Keep: userId, movieId, rating, drop : timestamp
	ratings_data = ratings_raw.filter(lambda line: line != ratings_header)\
	    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()


	movies_file = os.path.join(datasets_path, folder, movies_file)

	movies_raw = sc.textFile(movies_file)
	movies_header = movies_raw.take(1)[0]

	# Drop the genre column
	movies_data = movies_raw.filter(lambda line: line != movies_header) \
	    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()