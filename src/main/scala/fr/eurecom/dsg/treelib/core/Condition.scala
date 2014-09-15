package fr.eurecom.dsg.treelib.core

/**
 * This class is representative for the condition, eg. age < 12
 *
 * @param splitPoint The value of condition
 * @param positive The sign of comparing of this condition. Default value is true, it's mean 'less than' or belong to
 */
class Condition(val splitPoint: SplitPoint, val positive: Boolean = true) extends Serializable {

    /**
     * Check a value satisfy this condition or not
     */
    def check(value: Any) = {
        splitPoint.point match {
            case s: Set[String] => if (positive) s.contains(value.asInstanceOf[String]) else !s.contains(value.asInstanceOf[String])
            case d: Double => if (positive) (value.asInstanceOf[Double] < d) else !(value.asInstanceOf[Double] < d)
        }
    }
}
