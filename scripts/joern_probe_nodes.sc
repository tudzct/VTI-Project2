import io.shiftleft.semanticcpg.language.locationCreator

@main def main(): Unit = {
  println("calls=" + cpg.call.map(x => (x.name, x.code, x.location.filename)).take(10).l)
  println("ids=" + cpg.identifier.map(x => (x.name, x.code, x.location.filename)).take(10).l)
  println("controls=" + cpg.controlStructure.map(x => (x.code, x.controlStructureType, x.location.filename)).take(10).l)
}
