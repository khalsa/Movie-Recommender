
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.recommendation.Rating;

import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple4;
import scala.Tuple5;
import scala.Tuple6;
import scala.Tuple7;
import scala.Tuple8;

public class RecommenderAlgo {
public static void main(String args[]){
		
		
		//String inputPath = "file:///C:/Saurabh/SparkTest/MovieUserInput/u.data";
		//String movieInput = "file:///C:/Saurabh/SparkTest/MovieUserInput/u.item";
		
		//String inputPath = "file:///C:/Saurabh/SparkTest/MovieUserInput/u.data";
		
		String inputPath = "file:///C:/Saurabh/SparkTest/moviedata/ml-10M100K/ratings.dat";
		
		
		
		SparkConf conf = new SparkConf().setAppName("MovieRecommendation").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
		
		JavaRDD<String> data = sc.textFile(inputPath);
		
		JavaRDD<Rating> ratings = data.map(s -> {
			  String[] sarray = s.split("::");
			  //System.out.println("qerwwr " + sarray);
	          return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]), 
	                            Double.parseDouble(sarray[2]));
		});
		
		
		JavaPairRDD<Integer, Rating> calcRating2 = ratings.keyBy(f ->{ 
			return f.user();
		});
		
		JavaPairRDD<Integer, Rating> calcRating = ratings.keyBy(f ->{ 
			return f.user();
		});
		
		System.out.println("Size11111111111111 -------------------> " + calcRating2.collectAsMap().size());
		
		JavaPairRDD<Integer, Tuple2<Rating, Rating>> mergedUser = calcRating.join(calcRating2);
		
		System.out.println("Size222222222 -------------------> " + mergedUser.collectAsMap().size());
		
		JavaPairRDD<Integer, Tuple2<Rating, Rating>> filteredMergedUser = mergedUser.filter(f ->
		{ //System.out.println("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
			return f._2._1.product() < f._2._2().product();
			
		});
		
		
        JavaPairRDD<Tuple2<Integer, Integer>, Tuple3<Double, Double, Double>> intCalc = filteredMergedUser.mapToPair(d -> {
			
			Tuple2<Integer, Integer> key = new Tuple2(d._2._1.product(), d._2._2().product());
			Double ratingMultiply = d._2._1.rating()*d._2._2.rating();
			Double rating1 = d._2._1.rating();
			Double rating2 = d._2._2.rating();
			Double powRating1 = Math.pow(rating1, 2);
			Double powRating2 = Math.pow(rating2, 2);
			Tuple3<Double, Double, Double> stats = new Tuple3(ratingMultiply, powRating1, powRating2);
			
			Tuple2<Tuple2<Integer, Integer>, Tuple3<Double, Double, Double>> s = new Tuple2(key, stats);
			return s;
		});
        
        //System.out.println("Size33333333 -------------------> " + filteredMergedUser.collectAsMap().size());
		
		List<Tuple2<Tuple2<Integer, Integer>, Tuple3<Double, Double, Double>>> countList = new ArrayList<Tuple2<Tuple2<Integer, Integer>, Tuple3<Double, Double, Double>>>();
        countList = intCalc.collect();
        
      /*  for (Tuple2<Tuple2<Integer, Integer>, Tuple3<Double, Double, Double>> s : countList) {
            System.out.println("Key ----> " + s._1());
            System.out.println("Value ---> " + s._2());
            System.out.println("-----------------------------");
        }*/
		
        
        
        JavaPairRDD<Tuple2<Integer, Integer>, Tuple3<Double, Double, Double>> a = intCalc.groupByKey().mapToPair(f ->{
			 Iterator<Tuple3<Double, Double, Double>> i = f._2().iterator();
			 Double dotProduct = 0.0;
			 Double ratingSqSum = 0.0;
			 Double ratingSqSum2 = 0.0;
			 
			 while (i.hasNext()) {
				 Tuple3<Double, Double, Double> tempTuple = i.next();
				 dotProduct += tempTuple._1();
				 ratingSqSum += tempTuple._2();
				 ratingSqSum2 += tempTuple._3();
			}
			 
			 
			 Tuple3<Double, Double, Double> temp1 = new Tuple3(dotProduct, ratingSqSum, ratingSqSum2);
			 return new Tuple2<Tuple2<Integer, Integer>, Tuple3<Double, Double, Double>> (f._1(), temp1);
		});
        
        
		
		JavaRDD<Tuple3<Integer, Integer, Double>> similarities  = a.map(f -> {
			 Double dotProduct = f._2._1();
			 Double ratingSqSum = f._2._2();
			 Double ratingSqSum2 = f._2._3();
			 Double cosSim = cosineSimilarity(dotProduct, Math.sqrt(ratingSqSum),  Math.sqrt(ratingSqSum2));
			 	
			 return new Tuple3(f._1._1, f._1._2, cosSim);
			 
		});
		
		JavaRDD<Tuple3<Integer, Integer, Double>> filteredMovies  = similarities.filter(f -> {
			return f._1().equals(231) || f._2().equals(231);
			 
		}); 
		
		List<Tuple3<Integer, Integer, Double>> countList1 = new ArrayList<Tuple3<Integer, Integer, Double>>();
        countList1 = filteredMovies.collect();
        
        for(Tuple3<Integer, Integer, Double> s : countList1) {
            System.out.println("Key ----> " + s._1());
            System.out.println("Value ---> " + s._2());
            System.out.println("simmm ---> " + s._3());

            System.out.println("-----------------------------");
        }

        
        sc.stop();
	}

	public static Double cosineSimilarity(Double dotProduct, Double ratingNorm, Double rating2Norm){
		Double sim = dotProduct / (ratingNorm * rating2Norm);
		return sim;
	}
}
