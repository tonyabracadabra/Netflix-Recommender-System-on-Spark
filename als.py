from pyspark.mllib.recommendation import ALS
import math

seed = 5L
iterations = 10
regularization_parameter = 0.1
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1

def find_best_rank(training_RDD, ranks):
    for rank in ranks:
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                          lambda_=regularization_parameter)
        
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[err] = error
        err += 1
        print 'For rank %s the RMSE is %s' % (rank, error)
        if error < min_error:
            min_error = error
            best_rank = rank

    print 'The rank of the best model is %s' % best_rank

    return best_rank



def predict_ratings(self, user_and_movie_RDD):
    """Gets predictions for a given (userID, movieID) formatted RDD
    Returns: an RDD with format (movieTitle, movieRating, numRatings)
    """
    predicted_RDD = self.model.predictAll(user_and_movie_RDD)
    predicted_rating_RDD = predicted_RDD.map(lambda x: (x.product, x.rating))
    
    predicted_rating_title_and_count_RDD = \
        predicted_rating_RDD.join(self.movies_titles_RDD).join(self.movies_rating_counts_RDD)
    predicted_rating_title_and_count_RDD = \
        predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
    
    return predicted_rating_title_and_count_RDD
    
def add_ratings(self, ratings):
    """Add additional movie ratings in the format (user_id, movie_id, rating)
    """
    # Convert ratings to an RDD
    new_ratings_RDD = self.sc.parallelize(ratings)
    # Add new ratings to the existing ones
    self.ratings_RDD = self.ratings_RDD.union(new_ratings_RDD)
    # Re-compute movie ratings count
    self.__count_and_average_ratings()
    # Re-train the ALS model with the new ratings
    self.__train_model()
    
    return ratings