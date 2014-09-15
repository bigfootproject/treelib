package fr.eurecom.dsg.treelib.core

object Utility {
    /**
     * Parse a string to double
     */
    def parseDouble(s: String) = try { Some(s.toDouble) } catch { case _ => None }

    /**
     * Normalize a string.
     * Remove the double quotes if they appear at the beginning and the ending simultaneously.
     * 
     * @param s	string need to be normalized
     */
    def normalizeString(s: String) = {
        var s1 = s.trim
        val len = s1.length
        if (len <= 2 || s1 == "\"\"") s1
        else if (s1(0) == '\"' && s1(len - 1) == '\"')
            s1.substring(1, len - 1)
        else s1
    }

    /**
     * Convert a set into string with format {x,y,z}
     */
    def setToString(s: Set[String]): String = {
        val s1 = s.toString
        val len = s1.length
        if (len > 40)
            "{%s...}".format(s1.substring(4, 40))
        else
        	"{%s}".format(s1.substring(4, len-1))
    }

    def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
        val printWriter = new java.io.PrintWriter(f)
        try { op(printWriter) } finally { printWriter.close() }
    }

}
