
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
	public static void main(String args[]){
		System.out.println("sdfsfdsfd");
		
		String inputPath = "file:///C:/Saurabh/SparkTest/MovieUserInput/u.data";
		String movieInput = "file:///C:/Saurabh/SparkTest/MovieUserInput/u.item";
		
		SparkConf conf = new SparkConf().setAppName("MovieRecommendation").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
		
		JavaRDD<String> data = sc.textFile(inputPath);
		JavaRDD<String> movieData = sc.textFile(movieInput);
		
		JavaPairRDD<Integer, String> movies = movieData.mapToPair(s -> {
			 String[] sarray = s.split("\\|");
			 return new Tuple2<Integer, String>(Integer.parseInt(sarray[0]), sarray[1]);
		});
		
		Map<Integer, String> movieMap = movies.collectAsMap();
		
		
		JavaRDD<Rating> ratings = data.map(s -> {
			  String[] sarray = s.split("\\t");
	          return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]), 
	                            Double.parseDouble(sarray[2]));
		});
		
		int rank = 20;
	    int numIterations = 60;
	    MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);
	    
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
	    	return new Tuple2<Object, Double>(f._1, cosineSimilarity(tempMap.get(144), tempMap.get(f._1)));
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
