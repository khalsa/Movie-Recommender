package com.mr.test;

import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;

import scala.Tuple2;

public class KMeansMovieSimilarity {
	public static void main(String args[]){
		System.out.println("sdfsfdsfd");
		
		String inputPath = "file:///C:/Saurabh/SparkTest/ml-1m/ratings.dat";
		String movieInput = "file:///C:/Saurabh/SparkTest/ml-1m/movies.dat";
		
		String kMeansFormattedMovieInput = "file:///C:/Saurabh/SparkTest/ml-1m/movieFormatted.txt";

		
		SparkConf conf = new SparkConf().setAppName("MovieRecommendation").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
		
		JavaRDD<String> data = sc.textFile(inputPath);
		JavaRDD<String> movieData = sc.textFile(movieInput);
		
		    
	    JavaRDD<String> kMeansData = sc.textFile(kMeansFormattedMovieInput);
	    
	    JavaPairRDD<String, Vector> kMeansParsedData = kMeansData.mapToPair(s -> {
	    	String[] sarray = s.split("::");
	    	//System.out.println("dssadsasad ----------> " + sarray.length);
	    	
	    	double[] values = new double[18];
	     	int j = 0;
	    	 for (int i = 2; i < sarray.length; i++){
	        	  //System.out.println(" aaa " + sarray[i]);
		            values[j] = Double.parseDouble(sarray[i]);
		            j++;
	          }
	          return new Tuple2(sarray[0], Vectors.dense(values));
	    });
	    
	    Integer movieId = 296;
	    
	    int numClusters = 4;
	    int numIterationsKMeans = 50;
	    KMeansModel clusters = KMeans.train(kMeansParsedData.values().rdd(), numClusters, numIterationsKMeans);
	    
	    double WSSSE = clusters.computeCost(kMeansParsedData.values().rdd());
	    System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
	    
	    JavaPairRDD<String, Integer> movieClusteredData =  kMeansParsedData.mapToPair(f->{
	    	Integer b = clusters.predict(f._2());
	    		return new Tuple2(f._1(), b);
		    });
	    
	    Map<String, Integer> movieClusteredDataMap = movieClusteredData.collectAsMap();
		
		JavaPairRDD<Integer, String> movies = movieData.mapToPair(s -> {
			 String[] sarray = s.split("::");
			 return new Tuple2<Integer, String>(Integer.parseInt(sarray[0]), sarray[1]);
		});
		
		Map<Integer, String> movieMap = movies.collectAsMap();
		
		Integer clusterId = movieClusteredDataMap.get(String.valueOf(movieId));
		
		JavaRDD<Rating> ratings = data.map(s -> {
			  String[] sarray = s.split("::");
	          return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]), 
	                            Double.parseDouble(sarray[2]));
		});
		
		JavaRDD<Rating> filteredRatings = ratings.filter(s -> {
			  if(movieClusteredDataMap.get(String.valueOf(s.product())).equals(clusterId)){
				  return true;
			  }else{
				  return false;
			  }
		});
		
		int rank = 20;
	    int numIterations = 60;
	    MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(filteredRatings), rank, numIterations, 0.01);
	    
	  /*  List<Rating> ratedUser =  ratings.keyBy(f -> f.user()).lookup(269);
	    
	    ratedUser.stream().sorted(Comparator.comparingDouble(Rating::rating).reversed()).limit(10).forEach(f->{
	    	System.out.println("Movie Name :: " + movieMap.get(f.product()) +" Rating :: " + f.rating());
	    });
	    
	    Rating[] recMovie = model.recommendProducts(269, 30);
	    
	    	    
	    Set<Integer> acceptableNames = ratedUser.stream().map(Rating::product).collect(Collectors.toSet());
	    
	    Arrays.stream(recMovie).filter(e-> !acceptableNames.contains(e.product())).forEach(a->{
	    	System.out.println("RecommendedMovie Name :: " + movieMap.get(a.product()) +" Rating :: " + a.rating());
	    });
	  */  
	    //model.recommendProducts(userId, K)

	    

	    JavaPairRDD<Object, double[]> temp =  model.productFeatures().toJavaRDD().mapToPair(f->{
	    	return new Tuple2<Object, double[]>(f._1, f._2);
	    });
	    
	    Map<Object, double[]> tempMap = temp.collectAsMap();
	    
	    JavaPairRDD<Object, Double> cosSim =  model.productFeatures().toJavaRDD().mapToPair(f->{
	    	return new Tuple2<Object, Double>(f._1, cosineSimilarity(tempMap.get(movieId), tempMap.get(f._1)));
	    });
	    
	    //JavaPairRDD<Object, Double> cosSimSorted = (JavaPairRDD<Object, Double>) cosSim.mapToPair(x -> x.swap()).sortByKey(false).mapToPair(x -> x.swap()).take(10);

	    Map<Object, Double> cosineMap = cosSim.collectAsMap();
		
	    cosineMap.entrySet().stream()
        .sorted(Map.Entry.<Object, Double>comparingByValue().reversed()) 
        .limit(10) 
        .forEach(f->{
        	System.out.println("Movie Name :: " + movieMap.get(f.getKey()) + " value " + f.getValue());
        });
	    
	   
	    
	   //System.out.println(cosineSimilarity(tempMap.get(96),tempMap.get(96)));
	   
	   
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
