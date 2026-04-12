@main def main(fileName: String): Unit = {
  val m = cpg.method.filenameExact(fileName).nameNot("<global>")
  println("methodCount=" + m.size)
  println("methodNames=" + m.name.l.take(10))
  println("callCount=" + m.ast.isCall.size)
  println("identifierCount=" + m.ast.isIdentifier.size)
  println("controlCount=" + m.ast.isControlStructure.size)
  println("callNames=" + m.ast.isCall.name.l.take(10))
  println("identifierNames=" + m.ast.isIdentifier.name.l.take(10))
  println("controlCodes=" + m.ast.isControlStructure.code.l.take(10))
}
