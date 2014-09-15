package fr.eurecom.dsg.treelib.core

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._

// This class will load Feature Set from a file
class FeatureSet(metadata: List[Feature] = List[Feature]()) extends Serializable {

    // map a name of feature to its index
    private var mapNameToIndex: Map[String, Int] = Map[String, Int]() //withDefaultValue -1
    // we can not use withDefaulValue here. It will raise a NotSerializableExecptiopn
    // because of a bug in scala 2.9x. This bug is solved in 2.10
    
    var i = 0;
    metadata.foreach(f => {
        f.index = i
        mapNameToIndex = mapNameToIndex + (Utility.normalizeString(f.Name) -> i)
        i = i + 1
    })

    /**
     * Get index of a feature based on its name
     */
    def getIndex(name: String): Int = try { mapNameToIndex(name)}  catch { case _ : Throwable => -1 }
    
    def update(feature: Feature, index : Int)  = {  
        val oldFeature = data(index)
        mapNameToIndex = mapNameToIndex.-(oldFeature.Name)
        data = data.updated(index, feature)
        mapNameToIndex = mapNameToIndex.updated(feature.Name, index)
    }
    
    
    /**
     * List of feature info, we can see it like a map from index to name
     */
    private var rawData = List[Feature]()
    
    /**
     * Features information
     */
    var data = metadata
    
    /**
     * Number of features
     */
    lazy val numberOfFeature = data.length     
    
    override def toString() = {
      data.mkString(",\n")
    }
}
