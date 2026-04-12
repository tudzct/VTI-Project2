import io.shiftleft.semanticcpg.language.locationCreator

def clean(value: String): String = {
  value
    .replace("\\", "\\\\")
    .replace("\t", " ")
    .replace("\n", " ")
    .replace("\r", " ")
}

@main def main(): Unit = {
  cpg.call.foreach { node =>
    val filename = node.location.filename
    if (filename != null && filename != "<empty>") {
      println(s"${clean(filename)}\tCALL\t${clean(node.name)}\t${clean(node.code)}")
    }
  }
  cpg.identifier.foreach { node =>
    val filename = node.location.filename
    if (filename != null && filename != "<empty>") {
      println(s"${clean(filename)}\tIDENTIFIER\t${clean(node.name)}\t${clean(node.code)}")
    }
  }
  cpg.controlStructure.foreach { node =>
    val filename = node.location.filename
    if (filename != null && filename != "<empty>") {
      println(s"${clean(filename)}\tCONTROL_STRUCTURE\t${clean(node.controlStructureType)}\t${clean(node.code)}")
    }
  }
}
