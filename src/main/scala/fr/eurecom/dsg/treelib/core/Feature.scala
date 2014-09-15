package fr.eurecom.dsg.treelib.core

object FeatureType extends Enumeration {
    type FeatureType = Value
    val Numerical, Categorical = Value
}

import FeatureType._

/**
 * This class is representative for features
 * We have two feature categories : Numerical and Categorical
 * which are presented by NumericalFeature and CategoricalFeature, respectively
 *
 * @param Name	Name of feature, such as temperature, weather...
 * @param Type	Type of feature: Categorical or Numerical
 * @param index Index of this feature in the whole data set, based zero
 */
class Feature(var Name: String, var Type: FeatureType, var index: Int) extends Serializable {

    override def toString() = " (Index:" + index + " | Name: " + Name +
        " | Type: " + (if (Type == FeatureType.Numerical) "continuous" else "discrete") + ") ";

}

/**
 * Object instance FeatureInfo. It can be considered as a feature factory
 */
object Feature extends Serializable {
    def apply(Name: String, Type: FeatureType, idx: Int) = {
        new Feature(Utility.normalizeString(Name), Type, idx)
    }
    
    def unapply(feature: Feature) = Some(feature.Name, feature.Type, feature.index)
}

/**
 * We use this object to specify the type of a feature.
 * The index of feature will be updated later
 */
object as extends Serializable {
    def String(name : String) = Feature(name, FeatureType.Categorical, -1)
    def Number(name : String) = Feature(name, FeatureType.Numerical, -1)
}
