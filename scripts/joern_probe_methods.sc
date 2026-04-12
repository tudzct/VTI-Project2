@main def main(): Unit = {
  println(cpg.method.map(m => (m.name, m.fullName, m.filename)).take(20).l)
}
