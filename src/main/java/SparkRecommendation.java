
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.jblas.DoubleMatrix;

import scala.Tuple2;

public class SparkRecommendation {
	private static final String INPUT_RATING_PATH = "file:///C:/Saurabh/Spark/ml-1m/ratings.dat";
	private static final String INPUT_MOVIE_PATH = "file:///C:/Saurabh/Spark/ml-1m/movies.dat";
	private static final int TEST_MOVIE_ID = 2;
	
	public static void main(String args[]){
	    int rank = 20;
	    int numIterations = 20;
		
	    SparkConf conf = new SparkConf().setAppName("MovieRecommendation").setMaster("local");
            JavaSparkContext sc = new JavaSparkContext(conf);
		
	    JavaRDD<String> userRatingData = sc.textFile(INPUT_RATING_PATH);
	    JavaRDD<String> movieData = sc.textFile(INPUT_MOVIE_PATH);
		
	    JavaPairRDD<Integer, String> movies = movieData.mapToPair(s -> {
		String[] sarray = s.split("::");
		return new Tuple2<Integer, String>(Integer.parseInt(sarray[0]), sarray[1]);
	     });
		
	    Map<Integer, String> movieMap = movies.collectAsMap();
		
	    JavaRDD<Rating> ratingsRDD = userRatingData.map(s -> {
		  String[] sarray = s.split("::");
	          return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]), Double.parseDouble(sarray[2]));
	    });
		
	    MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratingsRDD), rank, numIterations, 0.01);

	    JavaPairRDD<Object, double[]> movieFeaturesRDDMap =  model.productFeatures().toJavaRDD().mapToPair(f->{
	    	return new Tuple2<Object, double[]>(f._1, f._2);
	    });
	    
	    Map<Object, double[]> movieFeaturesMap = movieFeaturesRDDMap.collectAsMap();
	    
	    JavaPairRDD<Object, Double> cosineSimilarity =  model.productFeatures().toJavaRDD().mapToPair(f->{
	    	return new Tuple2<Object, Double>(f._1, cosineSimilarity(movieFeaturesMap.get(TEST_MOVIE_ID), movieFeaturesMap.get(f._1)));
	    });
	    

	    Map<Object, Double> cosineMap = cosineSimilarity.collectAsMap();
		
	    //Sorting and showing top 10 match.
	    cosineMap.entrySet().stream()
           .sorted(Map.Entry.<Object, Double>comparingByValue().reversed()) 
           .limit(10) 
           .forEach(f->{
        	System.out.println("Movie Name :: " + movieMap.get(f.getKey()) + " similarity value :: " + f.getValue());
            });
	    
	    sc.stop();
	}
	
	public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
	    double dotProduct = 0.0;
	    double normA = 0.0;
	    double normB = 0.0;
	    for (int i = 0; i < vectorA.length; i++) {
	        dotProduct += vectorA[i] * vectorB[i];
	        normA += Math.pow(vectorA[i], 2);
	        normB += Math.pow(vectorB[i], 2);
	    }   
	    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
	}
}
