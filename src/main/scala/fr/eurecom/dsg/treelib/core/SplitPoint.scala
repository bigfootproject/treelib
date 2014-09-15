package fr.eurecom.dsg.treelib.core

/**
 * Representative of a split point
 *
 * @param index		index of the feature
 * @param point		the split point of this feature (it can be a Set, or a Double)
 * @param weight	the weight we get if we apply this splitting (often be a value of  an impurity function)
 */
class SplitPoint(val index: Int, var point: Any, var weight: Double) extends Serializable {
    override def toString = {
        var pointToString = (
                if (point.isInstanceOf[Array[_]]) point.asInstanceOf[Array[_]].mkString(",") 
                else if (point != null) point.toString
                else ""
                )
        "%d,%s,%f".format(
            index,
            (if (pointToString.length > 40)
                "%s...)".format(pointToString.substring(0, 40))
            else
                pointToString),
            weight)
    }
}
